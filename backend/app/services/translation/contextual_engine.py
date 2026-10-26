import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ChapterModel, NodeModel
from app.models.canonical import DocumentNode, NodeStatus
from app.services.translation.address_terms import AddressTermLedger
from app.services.translation.adaptive_chunker import AdaptiveSemanticChunker, SemanticChunk, TextSegment
from app.services.translation.context_assembler import ContextAssembler, TranslationContext, estimate_tokens
from app.services.translation.context_memory import BilingualContextItem, ChapterMemory, ChapterMemoryBuilder, RollingContextService
from app.services.translation.document_profiler import DocumentProfiler, DocumentTranslationProfile
from app.services.translation.entity_ledger import EntityLedgerService
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.json_parser import validate_translation_batch
from app.services.translation.node_policy import normalize_node_type
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.prompt_profiles import select_few_shots
from app.services.translation.provider_base import TranslationProvider
from app.services.translation.quality_gate import QualityGateResult, TranslationQualityGate
from app.services.translation.style_memory import StyleMemoryService
from app.services.translation.style_rules import StyleRuleService
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor


@dataclass
class EngineTelemetry:
    context_limit: int = 0
    system_tokens: int = 0
    document_tokens: int = 0
    chapter_tokens: int = 0
    rolling_tokens: int = 0
    glossary_tokens: int = 0
    soft_glossary_tokens: int = 0
    entity_tokens: int = 0
    few_shot_tokens: int = 0
    style_memory_tokens: int = 0
    few_shot_count: int = 0
    style_memory_hits: int = 0
    source_tokens: int = 0
    reserved_output: int = 0
    safety_margin: int = 0
    total_tokens: int = 0
    provider_calls: int = 0
    retries: int = 0
    context_trimmed: bool = False
    trim_steps: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class EngineNodeResult:
    node_id: str
    translated_text: str
    quality: QualityGateResult
    passed: bool
    attempts: int = 1
    error: str = ""
    telemetry: EngineTelemetry = field(default_factory=EngineTelemetry)


@dataclass
class EngineBatchResult:
    results: List[EngineNodeResult]
    mapping_error: bool = False
    telemetry: EngineTelemetry = field(default_factory=EngineTelemetry)


class ContextualTranslationEngine:
    def __init__(
        self,
        db: Session,
        project: Any,
        provider: TranslationProvider,
        config: TranslationConfig,
        locked_glossary: Optional[Dict[str, str]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        glossary_entries: Optional[List[Dict[str, Any]]] = None,
    ):
        self.db = db
        self.project = project
        self.project_id = str(project.id)
        self.provider = provider
        self.config = config
        legacy_locked_glossary = dict(locked_glossary or {})
        if glossary_entries is None:
            try:
                glossary_entries = GlossaryService.get_contextual_glossary(
                    db, self.project_id, domain=config.document_type,
                )
            except Exception:
                glossary_entries = []
        self.glossary_entries = list(glossary_entries or [])
        if self.glossary_entries:
            # Khi có glossary có domain, chỉ đưa hard terms đúng scope hiện tại vào prompt/gate.
            self.locked_glossary = {
                str(item.get("source_term", "")): str(
                    item.get("preferred_target") or item.get("target_term") or ""
                )
                for item in self.glossary_entries
                if item.get("source_term")
                and str(item.get("lock_level", "HARD") or "HARD").upper() == "HARD"
                and (item.get("preferred_target") or item.get("target_term"))
            }
        else:
            self.locked_glossary = legacy_locked_glossary
        self.glossary_validation_terms: Any = self.glossary_entries or self.locked_glossary
        self.soft_glossary = {
            str(item.get("source_term", "")): str(item.get("preferred_target", item.get("target_term", "")))
            for item in self.glossary_entries
            if str(item.get("lock_level", "HARD")).upper() == "SOFT"
            and item.get("source_term") and (item.get("preferred_target") or item.get("target_term"))
        }
        self.event_callback = event_callback
        self.address_ledger = AddressTermLedger.from_style_guide(config.style_guide, config.document_type)
        self.capabilities = provider.get_model_capabilities(config.model_name)
        self.effective_context_window = min(
            self.capabilities.context_window,
            self.capabilities.recommended_context_window,
        )
        self.provider.configure_context_window(self.effective_context_window)
        self.system_prompt = PromptBuilder.build_system_prompt(config)
        self.compact_system_prompt = PromptBuilder.build_system_prompt(config, include_optional_style_notes=False)
        self.cache_dir: Path = settings.PROJECTS_DIR / self.project_id / "cache"
        all_nodes = self.db.query(NodeModel).filter(
            NodeModel.project_id == self.project_id
        ).order_by(NodeModel.order_index).all()
        self.document_profile = DocumentProfiler.load_or_create(
            all_nodes,
            self.cache_dir,
            config.document_type,
            config.custom_instructions,
            provider,
            config.model_name,
        )
        self._chapter_memories: Dict[str, ChapterMemory] = {}

    def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.event_callback:
            self.event_callback(event_type, {"project_id": self.project_id, **data})

    def _translate_single_with_contract(
        self,
        text: str,
        system_prompt: str,
        glossary: Dict[str, str],
        temperature: float,
        user_prompt: str,
    ) -> str:
        """Gọi contract mới nhưng vẫn hỗ trợ provider mở rộng theo interface Phase 2 cũ."""
        try:
            return self.provider.translate_single(
                text=text,
                system_prompt=system_prompt,
                glossary_terms=glossary,
                model=self.config.model_name,
                temperature=temperature,
                user_prompt=user_prompt,
            )
        except TypeError as exc:
            if "user_prompt" not in str(exc):
                raise
            return self.provider.translate_single(
                text=text,
                system_prompt=f"{system_prompt}\n\n{user_prompt}",
                glossary_terms=glossary,
                model=self.config.model_name,
                temperature=temperature,
            )

    @staticmethod
    def canonical_node(node: NodeModel, content_override: Optional[str] = None) -> DocumentNode:
        try:
            status = NodeStatus(node.status)
        except ValueError:
            status = NodeStatus.PENDING
        return DocumentNode(
            id=node.id,
            type=normalize_node_type(node.node_type),
            content=node.content if content_override is None else content_override,
            status=status,
            order_index=node.order_index,
        )

    def chapter_memory(self, chapter: ChapterModel) -> ChapterMemory:
        if chapter.id not in self._chapter_memories:
            nodes = self.db.query(NodeModel).filter(
                NodeModel.chapter_id == chapter.id
            ).order_by(NodeModel.order_index).all()
            self._chapter_memories[chapter.id] = ChapterMemoryBuilder.load_or_create(
                chapter.id,
                chapter.title,
                nodes,
                self.cache_dir,
                self.locked_glossary,
                self.provider,
                self.config.model_name,
                self.config.document_type,
            )
        return self._chapter_memories[chapter.id]

    def _rolling_context(self, chapter: ChapterModel, nodes: List[DocumentNode], neighbors: bool) -> List[Any]:
        first = min(nodes, key=lambda item: item.order_index)
        if neighbors:
            db_node = self.db.query(NodeModel).filter(NodeModel.id == first.id).first()
            return RollingContextService.get_neighbors(self.db, db_node) if db_node else []
        return RollingContextService.get_previous(
            self.db, chapter.id, first.order_index, max_nodes=4, max_tokens=700,
        )

    def address_terms_for(self, source_text: str) -> AddressTermLedger:
        ledger = getattr(self, "address_ledger", None) or AddressTermLedger.from_style_guide(
            getattr(self.config, "style_guide", {}), getattr(self.config, "document_type", "GENERAL"),
        )
        return ledger.relevant(source_text)

    def project_style_rules(self) -> List[Any]:
        # Đọc theo request để quy tắc vừa được kích hoạt có hiệu lực ngay; lỗi DB không chặn việc dịch.
        try:
            return StyleRuleService.active_rules(self.db, self.project_id)
        except Exception:
            return []

    def assemble_context(
        self,
        chapter: ChapterModel,
        nodes: List[DocumentNode],
        prompt_kind: str,
        neighbors: bool = False,
        issues: Optional[List[Dict[str, Any]]] = None,
        continuity_context: Optional[Any] = None,
    ) -> TranslationContext:
        node_type = nodes[0].type.value if nodes else "paragraph"
        source_text = "\n".join(node.content or "" for node in nodes)
        few_shots = select_few_shots(
            self.config.document_type,
            self.config.translation_mode,
            node_type,
            limit=4 if str(getattr(self.config, "quality_tier", "")).upper() == "PUBLISHING" else 3,
            source_text=source_text,
        )
        style_examples = []
        if str(getattr(self.config, "quality_tier", "HIGH_QUALITY")).upper() != "FAST":
            try:
                style_examples = StyleMemoryService.retrieve_examples(
                    self.db,
                    self.project_id,
                    source_text,
                    domain=self.config.document_type,
                    document_type=self.config.document_type,
                    register=self.config.register,
                    translation_mode=self.config.translation_mode,
                    node_type=node_type,
                    limit=3,
                )
            except Exception:
                style_examples = []
        if prompt_kind == "batch":
            contract = PromptBuilder.BATCH_OUTPUT_CONTRACT
        elif prompt_kind == "segment":
            contract = PromptBuilder.SEGMENT_OUTPUT_CONTRACT
        else:
            contract = PromptBuilder.SINGLE_OUTPUT_CONTRACT
        if issues:
            contract += "\n" + "\n".join(
                f"{issue.get('code', 'QA')}: {issue.get('message', '')}" for issue in issues
            )
        context = ContextAssembler.assemble_context(
            nodes,
            self.document_profile,
            self.chapter_memory(chapter),
            [continuity_context] if continuity_context else self._rolling_context(chapter, nodes, neighbors),
            self.locked_glossary,
            self.capabilities,
            few_shots,
            system_prompt=self.system_prompt,
            output_contract=contract,
            prompt_kind=prompt_kind,
            compact_system_prompt=self.compact_system_prompt,
            soft_glossary=getattr(self, "soft_glossary", {}),
            glossary_entries=getattr(self, "glossary_entries", []),
            style_memory_examples=style_examples,
            continuity_context=continuity_context,
            address_terms=self.address_terms_for(source_text).prompt_lines(),
            style_rules=StyleRuleService.prompt_lines(self.project_style_rules()),
            # Đọc ledger theo từng request để quyết định mới có hiệu lực ngay trong cùng engine.
            entity_decisions=EntityLedgerService.relevant_decisions(
                self.db, self.project_id, source_text, self.locked_glossary,
            ),
        )
        if context.trim_steps:
            self._emit("TRANSLATION_CONTEXT_TRIMMED", {
                "chapter_id": chapter.id,
                "node_ids": [node.id for node in nodes],
                "trim_steps": context.trim_steps,
                "total_tokens": context.token_budget.total_estimated_tokens,
                "context_limit": context.token_budget.model_context_limit,
            })
        return context

    @staticmethod
    def telemetry_from_context(context: TranslationContext) -> EngineTelemetry:
        budget = context.token_budget
        return EngineTelemetry(
            context_limit=budget.model_context_limit,
            system_tokens=budget.system_tokens,
            document_tokens=budget.document_tokens,
            chapter_tokens=budget.chapter_tokens,
            rolling_tokens=budget.rolling_tokens,
            glossary_tokens=budget.glossary_tokens,
            soft_glossary_tokens=budget.soft_glossary_tokens,
            entity_tokens=budget.entity_tokens,
            few_shot_tokens=budget.few_shot_tokens,
            style_memory_tokens=budget.style_memory_tokens,
            few_shot_count=len(context.few_shots),
            style_memory_hits=len(context.style_memory_examples),
            source_tokens=budget.source_tokens,
            reserved_output=budget.reserved_output_tokens,
            safety_margin=budget.safety_margin_tokens,
            total_tokens=budget.total_estimated_tokens,
            context_trimmed=bool(context.trim_steps),
            trim_steps=list(context.trim_steps),
        )

    @staticmethod
    def _actual_request_tokens(context: TranslationContext, user_prompt: str) -> int:
        budget = context.token_budget
        return (
            estimate_tokens(context.system_prompt)
            + estimate_tokens(user_prompt)
            + budget.reserved_output_tokens
            + budget.safety_margin_tokens
        )

    def _request_fits(self, context: TranslationContext, user_prompt: str) -> bool:
        return self._actual_request_tokens(context, user_prompt) <= context.token_budget.model_context_limit

    def pack_next_chunk(self, chapter: ChapterModel, nodes: List[DocumentNode]) -> Optional[SemanticChunk]:
        current: List[DocumentNode] = []
        for node in nodes:
            if current and (
                current[0].type.value in {"table", "footnote"}
                or node.type.value in {"heading", "table", "footnote"}
            ):
                break
            candidate = current + [node]
            context = self.assemble_context(chapter, candidate, "batch")
            prompt = PromptBuilder.build_batch_prompt(candidate, context, chapter.title)
            if context.fits and self._request_fits(context, prompt):
                current = candidate
                continue
            if current:
                break
            current = [node]
            break
        if not current:
            return None
        context = self.assemble_context(chapter, current, "batch")
        return SemanticChunk(
            chunk_id=f"chunk_{current[0].order_index:05d}",
            nodes=current,
            estimated_tokens=context.token_budget.source_tokens,
        )

    def pack_nodes(self, chapter: ChapterModel, nodes: List[DocumentNode]) -> List[SemanticChunk]:
        remaining = list(nodes)
        chunks: List[SemanticChunk] = []
        while remaining:
            chunk = self.pack_next_chunk(chapter, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining = remaining[len(chunk.nodes):]
        return chunks

    def _empty_failure(self, node: DocumentNode, code: str, message: str, telemetry: Optional[EngineTelemetry] = None) -> EngineNodeResult:
        gate = TranslationQualityGate().validate(node.content, "", self.glossary_validation_terms)
        issues = list(gate.issues) + [{"code": code, "severity": "ERROR", "message": message}]
        failed_gate = QualityGateResult(False, True, 0.0, issues)
        return EngineNodeResult(node.id, "", failed_gate, False, error=message, telemetry=telemetry or EngineTelemetry())

    def translate_batch(self, chapter: ChapterModel, nodes: List[DocumentNode]) -> EngineBatchResult:
        context = self.assemble_context(chapter, nodes, "batch")
        if not context.fits:
            if len(nodes) == 1:
                return EngineBatchResult([self._translate_oversized_node(chapter, nodes[0])])
            return EngineBatchResult([
                self._empty_failure(node, "CONTEXT_BUDGET_EXCEEDED", "Batch cần được chia nhỏ hơn.") for node in nodes
            ])
        context.assert_within_budget()
        telemetry = self.telemetry_from_context(context)
        user_prompt = PromptBuilder.build_batch_prompt(nodes, context, chapter.title)
        if not self._request_fits(context, user_prompt):
            if len(nodes) == 1:
                return EngineBatchResult([self._translate_oversized_node(chapter, nodes[0])])
            return EngineBatchResult([
                self._empty_failure(node, "CONTEXT_BUDGET_EXCEEDED", "Prompt batch thực tế vượt context model.")
                for node in nodes
            ])
        telemetry.total_tokens = self._actual_request_tokens(context, user_prompt)
        started = time.perf_counter()
        try:
            telemetry.provider_calls += 1
            raw = self.provider.translate(
                blocks=[{"id": node.id, "text": node.content} for node in nodes],
                system_prompt=context.system_prompt,
                user_prompt=user_prompt,
                model=self.config.model_name,
                temperature=0.2,
            )
            parsed = validate_translation_batch(raw, [node.id for node in nodes])
        except Exception as exc:
            parsed = validate_translation_batch([], [node.id for node in nodes], str(exc))
        telemetry.latency_ms = (time.perf_counter() - started) * 1000
        mapping_error = bool(parsed.duplicate_ids or parsed.unknown_ids or parsed.raw_error)
        translated = {} if mapping_error else {item["node_id"]: item["text"] for item in parsed.translations}
        results: List[EngineNodeResult] = []
        for node in nodes:
            candidate = VietnamesePostProcessor.normalize_safely(translated.get(node.id, ""))
            gate = TranslationQualityGate().validate(node.content, candidate, self.glossary_validation_terms)
            if gate.passed:
                results.append(EngineNodeResult(node.id, candidate, gate, True, telemetry=telemetry))
                continue
            repair_issues = gate.issues or [{
                "code": "NODE_MAPPING_ERROR" if mapping_error else "MISSING_NODE_IDS",
                "severity": "ERROR",
                "message": "Provider không trả đúng node theo contract batch.",
            }]
            repaired = self.repair_node(chapter, node, repair_issues)
            repaired.telemetry.provider_calls += telemetry.provider_calls
            repaired.telemetry.latency_ms += telemetry.latency_ms
            results.append(repaired)
        return EngineBatchResult(results, mapping_error, telemetry)

    def translate_node(self, chapter: ChapterModel, node: DocumentNode, max_attempts: int = 2) -> EngineNodeResult:
        context = self.assemble_context(chapter, [node], "single", neighbors=True)
        if not context.fits:
            return self._translate_oversized_node(chapter, node)
        context.assert_within_budget()
        telemetry = self.telemetry_from_context(context)
        prompt = PromptBuilder.build_single_prompt(node, context, chapter.title)
        if not self._request_fits(context, prompt):
            return self._translate_oversized_node(chapter, node)
        telemetry.total_tokens = self._actual_request_tokens(context, prompt)
        started = time.perf_counter()
        try:
            telemetry.provider_calls += 1
            candidate = self._translate_single_with_contract(
                node.content, context.system_prompt, context.glossary, 0.15, prompt,
            )
        except Exception as exc:
            return self._empty_failure(node, "TRANSLATION_PROVIDER_ERROR", str(exc), telemetry)
        telemetry.latency_ms = (time.perf_counter() - started) * 1000
        candidate = VietnamesePostProcessor.normalize_safely(candidate)
        gate = TranslationQualityGate().validate(
            node.content, candidate, self.glossary_validation_terms, self._context_translations(context),
        )
        if gate.passed or max_attempts <= 1:
            return EngineNodeResult(node.id, candidate, gate, gate.passed, telemetry=telemetry)
        repaired = self.repair_node(chapter, node, gate.issues, max_attempts=max_attempts - 1)
        repaired.telemetry.provider_calls += telemetry.provider_calls
        repaired.telemetry.latency_ms += telemetry.latency_ms
        if not repaired.passed and self._has_foreign_script(repaired.quality.issues):
            return self._recover_foreign_script(node, repaired, context)
        return repaired

    @staticmethod
    def _context_translations(context: TranslationContext) -> List[str]:
        return [str(getattr(item, "translation", "") or "") for item in context.previous_context or []]

    @staticmethod
    def _has_foreign_script(issues: List[Dict[str, Any]]) -> bool:
        return any(issue.get("code") == "FOREIGN_SCRIPT_CONTAMINATION" for issue in issues)

    FOREIGN_SCRIPT_SYSTEM_PROMPT = (
        "Bạn là dịch giả Anh–Việt chuyên nghiệp. Chỉ viết tiếng Việt bằng chữ Quốc ngữ (chữ Latin có dấu). "
        "Tuyệt đối không dùng chữ Hán, tiếng Trung hay bất kỳ ký tự CJK nào. Giữ nguyên tên riêng, số liệu, "
        "mã, đường dẫn và thuật ngữ bắt buộc. Không thêm, không bớt ý."
    )
    CJK_RUN = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]+")

    def _recover_foreign_script(self, node: DocumentNode, failed: EngineNodeResult, context: TranslationContext) -> EngineNodeResult:
        """qwen2.5 thỉnh thoảng chuyển hẳn sang tiếng Trung với prompt dài; thử lại bằng prompt tiếng Việt ngắn.

        Bước 1 dịch lại cả đoạn với prompt tối giản. Nếu vẫn lọt chữ Hán, bước 2 chỉ dịch các cụm
        chữ Hán còn sót sang tiếng Việt rồi ghép lại. Kết quả vẫn phải qua quality gate như thường.
        """
        telemetry = failed.telemetry
        previous = self._context_translations(context)
        glossary_lines = "\n".join(f'- "{source}" -> "{target}"' for source, target in (context.glossary or {}).items())
        user_prompt = (
            (f"Thuật ngữ bắt buộc:\n{glossary_lines}\n\n" if glossary_lines else "")
            + f"Dịch đoạn tiếng Anh sau sang tiếng Việt, chỉ trả về bản dịch:\n\n{node.content}\n\n"
            + "Nhắc lại: chỉ dùng chữ Quốc ngữ, không dùng chữ Hán."
        )
        self._emit("TRANSLATION_FOREIGN_SCRIPT_RECOVERY", {"node_id": node.id, "step": "retranslate"})
        candidate = failed.translated_text
        try:
            telemetry.provider_calls += 1
            candidate = VietnamesePostProcessor.normalize_safely(self._translate_single_with_contract(
                node.content, self.FOREIGN_SCRIPT_SYSTEM_PROMPT, context.glossary, 0.0, user_prompt,
            ))
        except Exception:
            pass
        gate = TranslationQualityGate().validate(node.content, candidate, self.glossary_validation_terms, previous)
        if not gate.passed and self._has_foreign_script(gate.issues) and self.CJK_RUN.search(candidate):
            self._emit("TRANSLATION_FOREIGN_SCRIPT_RECOVERY", {"node_id": node.id, "step": "fragments"})
            pieces = []
            cursor = 0
            for match in self.CJK_RUN.finditer(candidate):
                pieces.append(candidate[cursor:match.start()])
                try:
                    telemetry.provider_calls += 1
                    pieces.append(self._translate_single_with_contract(
                        match.group(0),
                        self.FOREIGN_SCRIPT_SYSTEM_PROMPT,
                        context.glossary,
                        0.0,
                        f"Câu tiếng Anh gốc để tham khảo nghĩa:\n{node.content}\n\n"
                        f"Dịch cụm tiếng Trung sau sang tiếng Việt, chỉ trả về cụm đã dịch:\n{match.group(0)}",
                    ).strip())
                except Exception:
                    pieces.append(match.group(0))
                cursor = match.end()
            pieces.append(candidate[cursor:])
            candidate = VietnamesePostProcessor.normalize_safely(
                VietnamesePostProcessor.normalize_cjk_punctuation(" ".join(part.strip() for part in pieces if part.strip()))
            )
            gate = TranslationQualityGate().validate(node.content, candidate, self.glossary_validation_terms, previous)
        return EngineNodeResult(node.id, candidate, gate, gate.passed, attempts=failed.attempts + 1, telemetry=telemetry)

    def repair_node(
        self,
        chapter: ChapterModel,
        node: DocumentNode,
        issues: List[Dict[str, Any]],
        max_attempts: int = 2,
    ) -> EngineNodeResult:
        last = self._empty_failure(node, "REPAIR_FAILED", "Chưa có candidate repair hợp lệ.")
        for attempt in range(1, max_attempts + 1):
            context = self.assemble_context(chapter, [node], "repair", neighbors=True, issues=issues)
            if not context.fits:
                return self._empty_failure(node, "CONTEXT_BUDGET_EXCEEDED", "Repair prompt vượt context model.")
            context.assert_within_budget()
            telemetry = self.telemetry_from_context(context)
            telemetry.retries = attempt
            prompt = PromptBuilder.build_repair_prompt(node, context, issues, chapter.title)
            if not self._request_fits(context, prompt):
                return self._empty_failure(node, "CONTEXT_BUDGET_EXCEEDED", "Repair prompt thực tế vượt context model.")
            telemetry.total_tokens = self._actual_request_tokens(context, prompt)
            self._emit("TRANSLATION_RETRYING", {"node_id": node.id, "attempt": attempt, "issues": [issue.get("code") for issue in issues]})
            started = time.perf_counter()
            try:
                telemetry.provider_calls += 1
                candidate = self._translate_single_with_contract(
                    node.content,
                    context.system_prompt,
                    context.glossary,
                    0.1 if attempt == 1 else 0.0,
                    prompt,
                )
            except Exception as exc:
                last = self._empty_failure(node, "TRANSLATION_PROVIDER_ERROR", str(exc), telemetry)
                issues = last.quality.issues
                continue
            telemetry.latency_ms = (time.perf_counter() - started) * 1000
            candidate = VietnamesePostProcessor.normalize_safely(candidate)
            gate = TranslationQualityGate().validate(
                node.content, candidate, self.glossary_validation_terms, self._context_translations(context),
            )
            last = EngineNodeResult(node.id, candidate, gate, gate.passed, attempts=attempt + 1, telemetry=telemetry)
            if gate.passed:
                return last
            issues = gate.issues
        return last

    def preview_node(self, chapter: ChapterModel, node: DocumentNode) -> EngineNodeResult:
        return self.translate_node(chapter, node)

    def _translate_oversized_node(self, chapter: ChapterModel, node: DocumentNode) -> EngineNodeResult:
        initial_limit = max(64, min(self.capabilities.recommended_source_tokens, self.effective_context_window // 5))
        pending = AdaptiveSemanticChunker.split_with_layout(node.content, initial_limit)
        translated_parts: List[TextSegment] = []
        previous_segment: Optional[BilingualContextItem] = None
        total_telemetry = EngineTelemetry(context_limit=self.effective_context_window)
        index = 0
        while index < len(pending):
            part = pending[index]
            segment_node = node.model_copy(update={"content": part.text})
            context = self.assemble_context(
                chapter,
                [segment_node],
                "segment",
                neighbors=previous_segment is None,
                continuity_context=previous_segment,
            )
            if not context.fits:
                smaller_limit = min(
                    max(16, context.token_budget.available_source_tokens),
                    max(16, AdaptiveSemanticChunker.estimate_tokens(part.text) // 2),
                )
                smaller = AdaptiveSemanticChunker.split_with_layout(part.text, smaller_limit)
                if len(smaller) <= 1:
                    return self._empty_failure(node, "CONTEXT_BUDGET_EXCEEDED", "Một segment tối thiểu vẫn vượt context model.")
                smaller[-1] = TextSegment(smaller[-1].text, part.separator_after)
                pending[index:index + 1] = smaller
                continue
            context.assert_within_budget()
            telemetry = self.telemetry_from_context(context)
            prompt = PromptBuilder.build_segment_prompt(
                part.text, context, chapter.title, index + 1, len(pending),
            )
            if not self._request_fits(context, prompt):
                smaller_limit = max(16, AdaptiveSemanticChunker.estimate_tokens(part.text) // 2)
                smaller = AdaptiveSemanticChunker.split_with_layout(part.text, smaller_limit)
                if len(smaller) <= 1:
                    return self._empty_failure(node, "CONTEXT_BUDGET_EXCEEDED", "Segment thực tế vẫn vượt context model.")
                smaller[-1] = TextSegment(smaller[-1].text, part.separator_after)
                pending[index:index + 1] = smaller
                continue
            telemetry.total_tokens = self._actual_request_tokens(context, prompt)
            started = time.perf_counter()
            try:
                telemetry.provider_calls += 1
                candidate = self._translate_single_with_contract(
                    part.text, context.system_prompt, context.glossary, 0.1, prompt,
                )
            except Exception as exc:
                return self._empty_failure(node, "TRANSLATION_PROVIDER_ERROR", str(exc), telemetry)
            telemetry.latency_ms = (time.perf_counter() - started) * 1000
            normalized_candidate = VietnamesePostProcessor.normalize_safely(candidate)
            translated_parts.append(TextSegment(normalized_candidate, part.separator_after))
            previous_segment = BilingualContextItem(
                node_id=f"{node.id}:segment:{index + 1}",
                source=part.text,
                translation=normalized_candidate,
            )
            total_telemetry.provider_calls += telemetry.provider_calls
            total_telemetry.latency_ms += telemetry.latency_ms
            total_telemetry.total_tokens = max(total_telemetry.total_tokens, telemetry.total_tokens)
            index += 1
        combined = "".join(part.text + part.separator_after for part in translated_parts).rstrip()
        gate = TranslationQualityGate().validate(node.content, combined, self.glossary_validation_terms)
        return EngineNodeResult(node.id, combined, gate, gate.passed, attempts=max(1, len(translated_parts)), telemetry=total_telemetry)
