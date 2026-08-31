import json
import time
import uuid
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional

from app.db.models import NodeModel, QAIssueModel
from app.services.qa.vietnamese_naturalness_critic import (
    NATURALNESS_PASS_THRESHOLD,
    NATURALNESS_REWRITE_THRESHOLD,
    VietnameseNaturalnessCritic,
    VietnameseNaturalnessResult,
    naturalness_result_to_qa_issues,
)
from app.services.translation.entity_ledger import EntityLedgerService
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.semantic_assurance import SemanticAssuranceResult, SemanticAssuranceService
from app.services.qa.semantic_critic import semantic_signature
from app.services.translation.translation_commit import TranslationCommitService
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor


EDITORIAL_REWRITE_PROMPT_VERSION = "editorial-rewrite-v1"
MAX_EDITORIAL_REWRITE_ATTEMPTS = 1


class TranslationPublicationAssuranceService:
    """Điều phối semantic, naturalness và commit theo thứ tự fail-closed."""

    @staticmethod
    def naturalness_enabled(engine) -> bool:
        quality_profile = str(getattr(engine.config, "quality_profile", "BALANCED") or "BALANCED").upper()
        return quality_profile not in {"OFF", "DETERMINISTIC_ONLY"}

    @staticmethod
    def _semantic_node(engine, node):
        if hasattr(node, "type") and hasattr(node, "content") and not isinstance(node, NodeModel):
            return node
        return engine.canonical_node(node)

    @staticmethod
    def _previous_context(engine, chapter, node) -> List[Dict[str, str]]:
        try:
            values = engine._rolling_context(chapter, [TranslationPublicationAssuranceService._semantic_node(engine, node)], False)
        except Exception:
            values = []
        result = []
        for item in values or []:
            if isinstance(item, dict):
                result.append({
                    "source": str(item.get("source", "")),
                    "translation": str(item.get("translation", item.get("target", ""))),
                })
            else:
                result.append({
                    "source": str(getattr(item, "source", "")),
                    "translation": str(getattr(item, "translation", getattr(item, "target", ""))),
                })
        return result

    @classmethod
    def _naturalness_context(cls, engine, chapter, node) -> Dict[str, Any]:
        semantic_node = cls._semantic_node(engine, node)
        glossary = dict(getattr(engine, "locked_glossary", {}) or {})
        relevant_glossary = {
            source: target for source, target in glossary.items()
            if source.lower() in (semantic_node.content or "").lower()
        }
        return {
            "previous_context": cls._previous_context(engine, chapter, node),
            "glossary_terms": relevant_glossary,
            "entity_context": EntityLedgerService.relevant_decisions(
                engine.db, engine.project_id, semantic_node.content, glossary,
            ),
        }

    @classmethod
    def _review_naturalness(
        cls,
        engine,
        chapter,
        node,
        translated_text: str,
    ) -> VietnameseNaturalnessResult:
        config = engine.config
        try:
            context = cls._naturalness_context(engine, chapter, node)
            return VietnameseNaturalnessCritic.review(
                provider=engine.provider,
                source_text=node.content,
                translated_text=translated_text,
                document_type=str(getattr(config, "document_type", "GENERAL") or "GENERAL"),
                register=str(getattr(config, "register", "") or ""),
                sentence_style=str(getattr(config, "sentence_style", "MODERATE") or "MODERATE"),
                previous_context=context["previous_context"],
                glossary_terms=context["glossary_terms"],
                entity_context=context["entity_context"],
                model=str(
                    getattr(config, "naturalness_critic_model", None)
                    or getattr(config, "model_name", "")
                ),
            )
        except Exception as exc:
            # Lỗi khi dựng context cũng phải ERROR/fail-closed thay vì làm rơi cả worker.
            message = f"Naturalness critic context error: {exc}"
            return VietnameseNaturalnessResult(
                "ERROR", None,
                [{
                    "type": "VIETNAMESE_NATURALNESS_ERROR",
                    "severity": "ERROR",
                    "source_span": "",
                    "target_span": "",
                    "message": message,
                }],
                {}, message, 0, 0.0, 0, "",
            )

    @staticmethod
    def _thresholds(engine) -> tuple[float, float]:
        pass_threshold = float(getattr(engine.config, "naturalness_pass_threshold", NATURALNESS_PASS_THRESHOLD))
        rewrite_threshold = float(getattr(engine.config, "naturalness_rewrite_threshold", NATURALNESS_REWRITE_THRESHOLD))
        if not 0.0 <= pass_threshold <= 1.0:
            pass_threshold = NATURALNESS_PASS_THRESHOLD
        if not 0.0 <= rewrite_threshold < pass_threshold:
            rewrite_threshold = NATURALNESS_REWRITE_THRESHOLD
        return pass_threshold, rewrite_threshold

    @staticmethod
    def _rewrite_permitted(engine, result: VietnameseNaturalnessResult) -> bool:
        """Không rewrite chỉ vì văn phong formal hợp lệ ở domain nhạy cảm."""
        config = engine.config
        mode = str(getattr(config, "translation_mode", "NATURAL") or "NATURAL").upper()
        document_type = str(getattr(config, "document_type", "GENERAL") or "GENERAL").upper()
        protected = mode in {"FAITHFUL", "ACADEMIC", "TECHNICAL"} or document_type in {"LEGAL", "ACADEMIC", "TECHNICAL"}
        issue_types = {str(item.get("type", "")).upper() for item in result.issues}
        if protected and issue_types and issue_types.issubset({"REGISTER", "PASSIVE_VOICE", "SENTENCE_FLOW"}):
            return False
        return True

    @staticmethod
    def _attach_naturalness(
        result: SemanticAssuranceResult,
        naturalness: VietnameseNaturalnessResult,
        editorial_rewrite_count: int = 0,
        editorial_rewrite_success: bool = False,
        publication_status: str = "NEEDS_REVIEW",
    ) -> SemanticAssuranceResult:
        return replace(
            result,
            approved=result.approved if publication_status == "APPROVED" else False,
            naturalness_status=naturalness.status,
            naturalness_score=naturalness.score,
            naturalness_issues=list(naturalness.issues),
            naturalness_checks=dict(naturalness.checks),
            naturalness_critic_calls=naturalness.critic_calls,
            naturalness_latency_ms=naturalness.critic_latency_ms,
            editorial_rewrite_count=editorial_rewrite_count,
            editorial_rewrite_success=editorial_rewrite_success,
            publication_status=publication_status,
        )

    @staticmethod
    def _naturalness_payload(result: VietnameseNaturalnessResult, pass_threshold: float) -> Dict[str, Any]:
        return {
            "status": result.status,
            "score": result.score,
            "pass_threshold": pass_threshold,
            "issues": result.issues,
            "checks": result.checks,
            "signature": result.signature,
            "critic_request_tokens": result.critic_request_tokens,
            "critic_latency_ms": result.critic_latency_ms,
            "critic_calls": result.critic_calls,
        }

    @classmethod
    def _persist_naturalness_rejected(
        cls,
        engine,
        node,
        result: SemanticAssuranceResult,
        naturalness: VietnameseNaturalnessResult,
    ) -> None:
        """Lưu audit/QA issue nhưng tuyệt đối không lưu candidate chưa approved."""
        db_node = node if isinstance(node, NodeModel) else engine.db.query(NodeModel).filter(
            NodeModel.id == node.id,
        ).first()
        if not db_node:
            raise ValueError(f"Không tìm thấy node {node.id} để lưu naturalness review.")
        model_name = str(getattr(engine.config, "naturalness_critic_model", None) or getattr(engine.config, "model_name", ""))
        if result.approved:
            SemanticAssuranceService.persist_existing_review(
                engine.db, engine.project_id, db_node, result, model_name,
            )
        else:
            SemanticAssuranceService.persist_rejected(
                engine.db, engine.project_id, db_node, result, model_name,
            )
        for issue in naturalness_result_to_qa_issues(
            naturalness, db_node.content, result.translated_text,
        ):
            existing = engine.db.query(QAIssueModel).filter(
                QAIssueModel.project_id == engine.project_id,
                QAIssueModel.node_id == db_node.id,
                QAIssueModel.issue_type == issue["issue_type"],
                QAIssueModel.status == "OPEN",
            ).first()
            if existing:
                existing.severity = issue["severity"]
                existing.message = issue["message"]
                existing.source_snippet = issue["source_snippet"]
                existing.translation_snippet = issue["translation_snippet"]
                existing.suggested_fix = issue["suggested_fix"]
            else:
                engine.db.add(QAIssueModel(
                    id=str(uuid.uuid4()),
                    project_id=engine.project_id,
                    node_id=db_node.id,
                    issue_type=issue["issue_type"],
                    severity=issue["severity"],
                    message=issue["message"],
                    source_snippet=issue["source_snippet"],
                    translation_snippet=issue["translation_snippet"],
                    suggested_fix=issue["suggested_fix"],
                    status="OPEN",
                ))
        db_node.status = "NEEDS_REVIEW"
        engine.db.commit()

    @classmethod
    def assure_and_commit(
        cls,
        engine,
        chapter,
        node,
        translated_text: str,
        signature,
        model_name: str,
        created_by: str,
        instruction: Optional[str] = None,
        resolved_issue_ids: Optional[Iterable[str]] = None,
        latency_ms: float = 0.0,
        previous_repairs: int = 0,
    ) -> SemanticAssuranceResult:
        semantic_node = cls._semantic_node(engine, node)
        glossary = dict(getattr(engine, "locked_glossary", {}) or {})
        entity_context = EntityLedgerService.relevant_decisions(
            engine.db, engine.project_id, semantic_node.content, glossary,
        )
        locked_issues = EntityLedgerService.validate_locked(
            engine.db, engine.project_id, semantic_node.content, translated_text, glossary,
        )
        if locked_issues:
            signature_value = semantic_signature(
                semantic_node.content, translated_text, glossary, entity_context,
                getattr(engine.config, "semantic_critic_model", None) or engine.config.model_name,
                engine.config.document_type,
                SemanticAssuranceService.semantic_policy(
                    engine, getattr(engine.config, "semantic_max_repairs", 0),
                ),
            )
            rejected = SemanticAssuranceResult(
                translated_text, False, "FAIL", 1.0, "HIGH",
                [{"type": item["code"], "severity": item["severity"], "message": item["message"]} for item in locked_issues],
                signature_value, publication_status="NEEDS_REVIEW",
            )
            SemanticAssuranceService.persist_rejected(engine.db, engine.project_id, node, rejected, model_name)
            return rejected

        result = SemanticAssuranceService.evaluate_candidate(
            engine,
            chapter,
            semantic_node,
            translated_text,
            entity_context,
            previous_repairs=previous_repairs,
            max_repairs=getattr(engine.config, "semantic_max_repairs", 2),
        )
        if not result.approved:
            SemanticAssuranceService.persist_rejected(engine.db, engine.project_id, node, result, model_name)
            return result

        if not cls.naturalness_enabled(engine):
            approved = replace(result, publication_status="APPROVED")
            TranslationCommitService.commit_validated_translation(
                engine.db, engine.project_id, node, approved.translated_text, signature,
                glossary, model_name, created_by, instruction=instruction,
                resolved_issue_ids=resolved_issue_ids,
                semantic_result=approved.audit_payload(getattr(engine.config, "semantic_critic_model", None) or model_name),
                latency_ms=latency_ms,
            )
            return approved

        pass_threshold, rewrite_threshold = cls._thresholds(engine)
        naturalness = cls._review_naturalness(engine, chapter, node, result.translated_text)
        if naturalness.passed(pass_threshold):
            approved = cls._attach_naturalness(
                result, naturalness, publication_status="APPROVED",
            )
            TranslationCommitService.commit_validated_translation(
                engine.db, engine.project_id, node, approved.translated_text, signature,
                glossary, model_name, created_by, instruction=instruction,
                resolved_issue_ids=resolved_issue_ids,
                semantic_result=approved.audit_payload(getattr(engine.config, "semantic_critic_model", None) or model_name),
                latency_ms=latency_ms,
                naturalness_result=cls._naturalness_payload(naturalness, pass_threshold),
            )
            return approved

        if not naturalness.rewrite_allowed(rewrite_threshold, pass_threshold) or not cls._rewrite_permitted(engine, naturalness):
            rejected = cls._attach_naturalness(result, naturalness)
            cls._persist_naturalness_rejected(engine, node, rejected, naturalness)
            return rejected

        max_rewrites = max(0, min(MAX_EDITORIAL_REWRITE_ATTEMPTS, int(getattr(engine.config, "editorial_max_rewrites", 1))))
        if max_rewrites < 1:
            rejected = cls._attach_naturalness(result, naturalness)
            cls._persist_naturalness_rejected(engine, node, rejected, naturalness)
            return rejected

        rewritten = ""
        rewrite_error = "Editorial rewrite không trả về candidate hợp lệ."
        rewrite_latency = 0.0
        try:
            started = time.perf_counter()
            # Provider nhận ORIGINAL SOURCE và candidate hiện tại qua contract editorial riêng.
            context = cls._naturalness_context(engine, chapter, node)
            config = engine.config
            rewritten = engine.provider.editorial_rewrite(
                source_text=semantic_node.content,
                current_translation=result.translated_text,
                naturalness_issues=naturalness.issues,
                document_type=str(getattr(config, "document_type", "GENERAL") or "GENERAL"),
                register=str(getattr(config, "register", "") or ""),
                sentence_style=str(getattr(config, "sentence_style", "MODERATE") or "MODERATE"),
                glossary_terms=context["glossary_terms"],
                entity_context=context["entity_context"],
                model=str(getattr(config, "naturalness_critic_model", None) or getattr(config, "model_name", "")),
            )
            rewrite_latency = (time.perf_counter() - started) * 1000
        except NotImplementedError:
            try:
                started = time.perf_counter()
                instruction_text = (
                    f"PROMPT VERSION {EDITORIAL_REWRITE_PROMPT_VERSION}: Edit the Vietnamese, not the meaning. "
                    "Do not add, omit or change numbers, negation, modality, causality, scope, conditions, entities, "
                    "references or locked terminology. Naturalness issues: "
                    + json.dumps(naturalness.issues, ensure_ascii=False)
                )
                rewritten = engine.provider.revise(
                    semantic_node.content, result.translated_text, instruction_text,
                    model=getattr(config, "model_name", model_name),
                )
                rewrite_latency = (time.perf_counter() - started) * 1000
            except Exception as exc:
                rewritten = ""
                rewrite_latency = 0.0
                rewrite_error = str(exc)
        except Exception as exc:
            rewritten = ""
            rewrite_latency = 0.0
            rewrite_error = str(exc)

        if not isinstance(rewritten, str):
            # Provider trả sai contract thì chuyển thẳng sang NEEDS_REVIEW, không để lỗi kiểu dữ liệu lọt qua.
            rewrite_error = "Editorial rewrite phải trả về plain text tiếng Việt."
            rewritten = ""
        rewritten = VietnamesePostProcessor.normalize_safely(rewritten or "")
        if not rewritten:
            error_issue = VietnameseNaturalnessResult(
                "ERROR", None,
                [{
                    "type": "VIETNAMESE_NATURALNESS_ERROR",
                    "severity": "ERROR",
                    "target_span": "",
                    "message": rewrite_error,
                }],
                {}, rewrite_error,
                0, rewrite_latency, 1, naturalness.signature,
            )
            rejected = cls._attach_naturalness(result, error_issue, 1, False)
            cls._persist_naturalness_rejected(engine, node, rejected, error_issue)
            return rejected

        # Dù deterministic recheck fail, vẫn chạy đủ hai critic theo contract P0;
        # kết quả cuối cùng chỉ được APPROVED khi cả ba tầng đều đạt.
        rewritten_quality = TranslationQualityGate().validate(semantic_node.content, rewritten, glossary)
        rewritten_entity_issues = EntityLedgerService.validate_locked(
            engine.db, engine.project_id, semantic_node.content, rewritten, glossary,
        )
        semantic_recheck = SemanticAssuranceService.evaluate_candidate(
            engine,
            chapter,
            semantic_node,
            rewritten,
            entity_context,
            previous_repairs=previous_repairs,
            max_repairs=0,
            force_critic=True,
        )
        naturalness_recheck = cls._review_naturalness(engine, chapter, node, rewritten)
        total_naturalness_calls = naturalness.critic_calls + naturalness_recheck.critic_calls
        total_naturalness_latency = naturalness.critic_latency_ms + naturalness_recheck.critic_latency_ms
        naturalness_recheck = replace(
            naturalness_recheck,
            critic_calls=total_naturalness_calls,
            critic_latency_ms=total_naturalness_latency,
        )
        if rewritten_entity_issues:
            # Entity lock là invariant riêng, không phụ thuộc semantic critic có nhận ra hay không.
            entity_errors = [
                {"type": issue["code"], "severity": issue["severity"], "message": issue["message"]}
                for issue in rewritten_entity_issues
            ]
            semantic_recheck = replace(
                semantic_recheck,
                approved=False,
                status="FAIL",
                errors=list(semantic_recheck.errors) + entity_errors,
            )
        rewrite_success = (
            rewritten_quality.passed
            and not rewritten_entity_issues
            and semantic_recheck.approved
            and naturalness_recheck.passed(pass_threshold)
        )
        final_result = cls._attach_naturalness(
            semantic_recheck,
            naturalness_recheck,
            editorial_rewrite_count=1,
            editorial_rewrite_success=rewrite_success,
            publication_status="APPROVED" if rewrite_success else "NEEDS_REVIEW",
        )
        if not rewrite_success:
            cls._persist_naturalness_rejected(engine, node, final_result, naturalness_recheck)
            return final_result

        TranslationCommitService.commit_validated_translation(
            engine.db, engine.project_id, node, final_result.translated_text, signature,
            glossary, model_name, created_by, instruction=instruction,
            resolved_issue_ids=resolved_issue_ids,
            semantic_result=final_result.audit_payload(getattr(engine.config, "semantic_critic_model", None) or model_name),
            latency_ms=latency_ms + rewrite_latency,
            naturalness_result=cls._naturalness_payload(naturalness_recheck, pass_threshold),
        )
        return final_result
