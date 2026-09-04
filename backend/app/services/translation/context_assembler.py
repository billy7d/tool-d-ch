import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.translation.context_memory import BilingualContextItem, ChapterMemory
from app.services.translation.document_profiler import DocumentTranslationProfile
from app.services.translation.model_capabilities import ModelCapabilities
from app.services.translation.term_matcher import TermMatcher


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    punctuation = sum(1 for char in text if char in ".,;:!?()[]{}\"'")
    non_ascii = sum(1 for char in text if ord(char) > 127)
    return max(1, int(words * 1.22 + punctuation * 0.22 + non_ascii * 0.08 + 8))


class ContextBudgetExceeded(RuntimeError):
    pass


# Chính sách này là hợp đồng chung cho NATURAL/BALANCED và được giữ độc lập
# với cách provider cụ thể dựng prompt.
NATURAL_BALANCED_CONTEXT_PRIORITY: Tuple[str, ...] = (
    "system_contract",
    "source",
    "hard_glossary_and_entities",
    "immediate_previous_bilingual",
    "user_style_memory",
    "curated_few_shot",
    "chapter_memory",
    "distant_rolling_context",
    "broad_document_memory",
)

CONTEXT_TRIM_ORDER: Tuple[str, ...] = (
    "document_memory_compact",
    "chapter_memory_compact",
    "rolling_context_oldest",
    "entity_decisions_optional",
    "soft_glossary_optional",
    "style_memory_oldest",
    "few_shot_reduce",
    "few_shot_drop_emergency",
)


@dataclass(frozen=True)
class ContextTokenBudget:
    model_context_limit: int
    system_tokens: int
    document_tokens: int
    chapter_tokens: int
    rolling_tokens: int
    glossary_tokens: int
    soft_glossary_tokens: int
    entity_tokens: int
    few_shot_tokens: int
    style_memory_tokens: int
    contract_tokens: int
    context_tokens: int
    source_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    total_estimated_tokens: int
    available_source_tokens: int
    fits_context_window: bool
    fits_source_budget: bool
    fits: bool

    @property
    def context_window(self) -> int:
        return self.model_context_limit

    @property
    def reserved_output(self) -> int:
        return self.reserved_output_tokens

    @property
    def safety_margin(self) -> int:
        return self.safety_margin_tokens

    @property
    def fixed_prompt_tokens(self) -> int:
        return self.system_tokens + self.contract_tokens

    @property
    def source_budget(self) -> int:
        return self.available_source_tokens

    @property
    def total_estimated_input(self) -> int:
        return self.system_tokens + self.context_tokens + self.source_tokens + self.contract_tokens

    def assert_within_budget(self) -> None:
        if not self.fits_context_window:
            raise ContextBudgetExceeded(
                f"Ngữ cảnh ước tính {self.total_estimated_tokens} token vượt giới hạn {self.model_context_limit}."
            )
        if not self.fits_source_budget:
            raise ContextBudgetExceeded(
                f"Nguồn ước tính {self.source_tokens} token vượt ngân sách nguồn "
                f"{self.available_source_tokens}."
            )


@dataclass
class TranslationContext:
    document_memory: str = ""
    chapter_memory: str = ""
    previous_context: List[BilingualContextItem] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    soft_glossary: Dict[str, str] = field(default_factory=dict)
    glossary_details: List[dict] = field(default_factory=list)
    soft_glossary_details: List[dict] = field(default_factory=list)
    entity_decisions: Dict[str, str] = field(default_factory=dict)
    few_shots: List[dict] = field(default_factory=list)
    style_memory_examples: List[dict] = field(default_factory=list)
    continuity_context: Optional[BilingualContextItem] = None
    context_priority: Tuple[str, ...] = NATURAL_BALANCED_CONTEXT_PRIORITY
    token_budget: Optional[ContextTokenBudget] = None
    system_prompt: str = ""
    trim_steps: List[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return bool(self.token_budget and self.token_budget.fits)

    def assert_within_budget(self) -> None:
        if not self.token_budget:
            raise ContextBudgetExceeded("Thiếu token budget cho request dịch.")
        self.token_budget.assert_within_budget()


class ContextAssembler:
    MIN_SOURCE_BUDGET = 64
    SAFETY_MARGIN_MIN = 220
    CONTEXT_LABEL_OVERHEAD = 72

    @staticmethod
    def filter_relevant_glossary(source_text: str, glossary: Dict[str, str]) -> Dict[str, str]:
        relevant: Dict[str, str] = {}
        for source, target in (glossary or {}).items():
            if TermMatcher.contains(source_text, source):
                relevant[source] = target
        return relevant

    @staticmethod
    def _source_payload(nodes: List[Any], prompt_kind: str) -> str:
        if prompt_kind == "batch":
            blocks = []
            for node in nodes:
                node_type = getattr(getattr(node, "type", None), "value", getattr(node, "node_type", "paragraph"))
                blocks.append({"node_id": getattr(node, "id", ""), "type": node_type, "text": getattr(node, "content", "") or ""})
            return json.dumps(blocks, ensure_ascii=False, indent=2)
        return "\n".join(str(getattr(node, "content", "") or "") for node in nodes)

    @staticmethod
    def _compact_chapter(memory: ChapterMemory, level: int) -> str:
        if level == 1:
            value = {
                "summary": memory.summary[:1200], "entities": memory.entities[:10],
                "terminology": memory.terminology[:10], "pronoun_notes": memory.pronoun_notes[:8],
                "tone": memory.tone[:300], "style_notes": memory.style_notes[:5],
            }
        else:
            value = {
                "summary": memory.summary[:500], "entities": memory.entities[:5],
                "terminology": memory.terminology[:5], "pronoun_notes": memory.pronoun_notes[:3],
            }
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _compact_document(profile: DocumentTranslationProfile, level: int) -> str:
        source = profile.to_source_context()
        if level == 1:
            value = {
                "document_type": source["document_type"], "source_domain": source["source_domain"],
                "source_audience": source["source_audience"], "source_tone": source["source_tone"],
                "source_register": source["source_register"],
                "style_observations": source["style_observations"][:5],
                "source_summary": source["source_summary"][:900],
            }
        else:
            value = {
                "source_domain": source["source_domain"], "source_tone": source["source_tone"],
                "source_register": source["source_register"], "source_summary": source["source_summary"][:400],
            }
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def assemble_context(
        cls,
        nodes: List[Any],
        document_profile: DocumentTranslationProfile,
        chapter_memory: ChapterMemory,
        rolling_context: List[BilingualContextItem],
        glossary: Dict[str, str],
        model_capabilities: ModelCapabilities,
        few_shots: Optional[List[dict]] = None,
        system_prompt: str = "",
        output_contract: str = "",
        prompt_kind: str = "batch",
        compact_system_prompt: Optional[str] = None,
        entity_decisions: Optional[Dict[str, str]] = None,
        soft_glossary: Optional[Dict[str, str]] = None,
        style_memory_examples: Optional[List[dict]] = None,
        continuity_context: Optional[BilingualContextItem] = None,
        glossary_entries: Optional[List[dict]] = None,
    ) -> TranslationContext:
        source_text = "\n".join(str(getattr(node, "content", "") or "") for node in nodes)
        source_payload = cls._source_payload(nodes, prompt_kind)
        document_memory = json.dumps(document_profile.to_source_context(), ensure_ascii=False, separators=(",", ":"))
        chapter_text = json.dumps(chapter_memory.to_dict(), ensure_ascii=False, separators=(",", ":"))
        relevant_glossary = cls.filter_relevant_glossary(source_text, glossary)
        relevant_soft_glossary = cls.filter_relevant_glossary(source_text, soft_glossary or {})
        relevant_glossary_details: List[dict] = []
        relevant_soft_details: List[dict] = []
        for entry in glossary_entries or []:
            source_term = str(entry.get("source_term", entry.get("source", "")) or "").strip()
            preferred = str(
                entry.get("preferred_target") or entry.get("target_term") or entry.get("target") or ""
            ).strip()
            if not source_term or not TermMatcher.contains(source_text, source_term) or not preferred:
                continue
            normalized_entry = dict(entry)
            normalized_entry["source_term"] = source_term
            normalized_entry["preferred_target"] = preferred
            normalized_entry["allowed_variants"] = list(entry.get("allowed_variants") or [])
            level = str(entry.get("lock_level", "HARD") or "HARD").upper()
            if level == "SOFT":
                relevant_soft_details.append(normalized_entry)
                relevant_soft_glossary[source_term] = preferred
            else:
                relevant_glossary_details.append(normalized_entry)
                relevant_glossary[source_term] = preferred
        if not relevant_glossary_details:
            relevant_glossary_details = [
                {"source_term": source, "preferred_target": target, "lock_level": "HARD", "allowed_variants": []}
                for source, target in relevant_glossary.items()
            ]
        if not relevant_soft_details:
            relevant_soft_details = [
                {"source_term": source, "preferred_target": target, "lock_level": "SOFT", "allowed_variants": []}
                for source, target in relevant_soft_glossary.items()
            ]
        relevant_entities = cls.filter_relevant_glossary(source_text, entity_decisions or {})
        selected_shots = list(few_shots or [])
        selected_style = list(style_memory_examples or [])
        previous = list(rolling_context)
        if continuity_context:
            # Segment continuity là context immediate bắt buộc; không giữ lịch sử segment cũ.
            previous = [continuity_context]
        selected_system = system_prompt
        trim_steps: List[str] = []
        context_limit = min(model_capabilities.context_window, model_capabilities.recommended_context_window)
        source_tokens = estimate_tokens(source_payload)
        reserved_output = max(320, min(1400, int(source_tokens * 1.35) + 128))
        safety_margin = max(cls.SAFETY_MARGIN_MIN, int(context_limit * 0.06))

        # Payload few-shot do caller cũ truyền vào không có metadata curated và
        # không được chiếm phần lớn cửa sổ context. Các ví dụ curated đều nhỏ.
        oversized_external_shots = [
            item for item in selected_shots
            if not item.get("example_id")
            and estimate_tokens(json.dumps(item, ensure_ascii=False)) > max(256, context_limit // 6)
        ]
        if oversized_external_shots:
            selected_shots = [item for item in selected_shots if item not in oversized_external_shots]
            trim_steps.append("few_shot_drop_emergency")

        def components() -> Dict[str, int]:
            return {
                "system": estimate_tokens(selected_system),
                "document": estimate_tokens(document_memory),
                "chapter": estimate_tokens(chapter_text),
                "rolling": estimate_tokens(json.dumps([item.__dict__ for item in previous], ensure_ascii=False)),
                "glossary": estimate_tokens(json.dumps(relevant_glossary_details, ensure_ascii=False)),
                "soft_glossary": estimate_tokens(json.dumps(relevant_soft_details, ensure_ascii=False)),
                "entities": estimate_tokens(json.dumps(relevant_entities, ensure_ascii=False)),
                "few_shot": estimate_tokens(json.dumps(selected_shots, ensure_ascii=False)),
                "style_memory": estimate_tokens(json.dumps(selected_style, ensure_ascii=False)),
                "contract": estimate_tokens(output_contract) + cls.CONTEXT_LABEL_OVERHEAD,
            }

        def total() -> int:
            values = components()
            return sum(values.values()) + source_tokens + reserved_output + safety_margin

        # Broad document memory được compact trước vì là lớp xa nhất.
        if total() > context_limit:
            document_memory = cls._compact_document(document_profile, 1)
            trim_steps.append("document_memory_compact_1")
        if total() > context_limit:
            document_memory = cls._compact_document(document_profile, 2)
            trim_steps.append("document_memory_compact_2")
        # Chapter memory vẫn giữ summary/thuật ngữ cốt lõi nhưng giảm dần độ chi tiết.
        if total() > context_limit:
            chapter_text = cls._compact_chapter(chapter_memory, 1)
            trim_steps.append("chapter_memory_compact_1")
        if total() > context_limit:
            chapter_text = cls._compact_chapter(chapter_memory, 2)
            trim_steps.append("chapter_memory_compact_2")
        # Không xóa bilingual context cuối cùng: nó là điểm nối source -> Vietnamese
        # đã duyệt gần nhất, đặc biệt quan trọng với đại từ và mạch văn.
        while total() > context_limit and len(previous) > 1:
            previous.pop(0)
            if "rolling_context_oldest" not in trim_steps:
                trim_steps.append("rolling_context_oldest")
        # Entity decision là context tùy chọn; hard glossary ở trên vẫn nguyên vẹn.
        while total() > context_limit and relevant_entities:
            relevant_entities.pop(next(reversed(relevant_entities)))
            if "entity_decisions_optional" not in trim_steps:
                trim_steps.append("entity_decisions_optional")
        # SOFT glossary cung cấp gợi ý, không được che lấp semantic contract.
        while total() > context_limit and relevant_soft_glossary:
            removed_source = next(reversed(relevant_soft_glossary))
            relevant_soft_glossary.pop(removed_source)
            relevant_soft_details = [
                item for item in relevant_soft_details
                if item.get("source_term") != removed_source
            ]
            if "soft_glossary_optional" not in trim_steps:
                trim_steps.append("soft_glossary_optional")
        # User style quan trọng hơn curated examples, nên chỉ bỏ các mẫu xa/cũ.
        while total() > context_limit and len(selected_style) > 1:
            selected_style.pop(0)
            if "style_memory_oldest" not in trim_steps:
                trim_steps.append("style_memory_oldest")
        while total() > context_limit and len(selected_shots) > 1:
            selected_shots.pop()
            if "few_shot_reduce" not in trim_steps:
                trim_steps.append("few_shot_reduce")
        # Few-shot caller cũ có thể truyền một payload khổng lồ. Không để payload
        # đó phá budget; curated examples luôn nhỏ và vẫn được giữ lại.
        if total() > context_limit and selected_shots:
            only_shot_is_oversized = len(selected_shots) == 1 and estimate_tokens(
                json.dumps(selected_shots[0], ensure_ascii=False)
            ) > max(256, context_limit // 6)
            if only_shot_is_oversized:
                selected_shots = []
                trim_steps.append("few_shot_drop_emergency")
        if total() > context_limit and len(selected_style) == 1:
            # Style memory cũng không được phép đẩy source/contract ra ngoài cửa sổ.
            if estimate_tokens(json.dumps(selected_style[0], ensure_ascii=False)) > max(256, context_limit // 6):
                selected_style = []
                trim_steps.append("style_memory_drop_emergency")
        if total() > context_limit and compact_system_prompt and compact_system_prompt != selected_system:
            selected_system = compact_system_prompt
            trim_steps.append("optional_style_notes")

        values = components()
        context_tokens = (
            values["document"] + values["chapter"] + values["rolling"]
            + values["glossary"] + values["soft_glossary"] + values["entities"]
            + values["few_shot"] + values["style_memory"]
        )
        non_source = values["system"] + context_tokens + values["contract"] + reserved_output + safety_margin
        available_source = min(model_capabilities.recommended_source_tokens, max(0, context_limit - non_source))
        total_estimated = non_source + source_tokens
        fits_context_window = total_estimated <= context_limit
        fits_source_budget = source_tokens <= available_source
        budget = ContextTokenBudget(
            model_context_limit=context_limit,
            system_tokens=values["system"],
            document_tokens=values["document"],
            chapter_tokens=values["chapter"],
            rolling_tokens=values["rolling"],
            glossary_tokens=values["glossary"],
            soft_glossary_tokens=values["soft_glossary"],
            entity_tokens=values["entities"],
            few_shot_tokens=values["few_shot"],
            style_memory_tokens=values["style_memory"],
            contract_tokens=values["contract"],
            context_tokens=context_tokens,
            source_tokens=source_tokens,
            reserved_output_tokens=reserved_output,
            safety_margin_tokens=safety_margin,
            total_estimated_tokens=total_estimated,
            available_source_tokens=available_source,
            fits_context_window=fits_context_window,
            fits_source_budget=fits_source_budget,
            fits=fits_context_window and fits_source_budget,
        )
        return TranslationContext(
            document_memory=document_memory,
            chapter_memory=chapter_text,
            previous_context=previous,
            glossary=relevant_glossary,
            soft_glossary=relevant_soft_glossary,
            glossary_details=relevant_glossary_details,
            soft_glossary_details=relevant_soft_details,
            entity_decisions=relevant_entities,
            few_shots=selected_shots,
            style_memory_examples=selected_style,
            continuity_context=continuity_context,
            context_priority=NATURAL_BALANCED_CONTEXT_PRIORITY,
            token_budget=budget,
            system_prompt=selected_system,
            trim_steps=trim_steps,
        )
