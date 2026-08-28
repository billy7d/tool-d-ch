from dataclasses import dataclass
from typing import Any, Dict

import uuid

from app.db.models import NodeModel, QAIssueModel, SemanticReviewModel
from app.services.qa.semantic_critic import (
    SEMANTIC_CRITIC_PROMPT_VERSION,
    SemanticCritic,
    semantic_signature,
)
from app.services.translation.semantic_risk import SemanticRiskScorer
from app.services.translation.entity_ledger import EntityLedgerService
from app.services.translation.translation_commit import TranslationCommitService
from app.services.translation.term_matcher import TermMatcher


SEMANTIC_REPAIR_PROMPT_VERSION = "semantic-repair-v1"


@dataclass
class SemanticAssuranceResult:
    translated_text: str
    approved: bool
    status: str
    risk_score: float
    risk_level: str
    errors: list[Dict[str, Any]]
    signature: str
    score: float | None = None
    repair_attempts: int = 0
    critic_calls: int = 0
    critic_request_tokens: int = 0
    critic_latency_ms: float = 0.0
    from_cache: bool = False

    def audit_payload(self, model_name: str) -> Dict[str, Any]:
        return {
            "signature": self.signature, "risk_score": self.risk_score, "risk_level": self.risk_level,
            "status": self.status, "score": self.score, "errors": self.errors,
            "model_name": model_name, "prompt_version": SEMANTIC_CRITIC_PROMPT_VERSION,
            "critic_calls": self.critic_calls, "critic_request_tokens": self.critic_request_tokens,
            "critic_latency_ms": self.critic_latency_ms,
        }


class SemanticAssuranceService:
    @classmethod
    def invalidate_for_nodes(cls, db, project_id: str, node_ids) -> int:
        ids = list(dict.fromkeys(node_ids or []))
        if not ids:
            return 0
        return db.query(SemanticReviewModel).filter(
            SemanticReviewModel.project_id == project_id,
            SemanticReviewModel.node_id.in_(ids),
            SemanticReviewModel.is_stale.is_(False),
        ).update({"is_stale": True}, synchronize_session="fetch")

    @classmethod
    def invalidate_for_source_term(cls, db, project_id: str, source_term: str) -> int:
        nodes = db.query(NodeModel).filter_by(project_id=project_id).all()
        return cls.invalidate_for_nodes(
            db, project_id,
            [node.id for node in nodes if TermMatcher.contains(node.content or "", source_term)],
        )

    @classmethod
    def assure_and_commit(
        cls, engine, chapter, node, translated_text: str, signature, model_name: str,
        created_by: str, instruction: str | None = None,
        resolved_issue_ids=None, latency_ms: float = 0.0, previous_repairs: int = 0,
    ) -> SemanticAssuranceResult:
        entity_context = EntityLedgerService.relevant_decisions(
            engine.db, engine.project_id, node.content, engine.locked_glossary,
        )
        locked_issues = EntityLedgerService.validate_locked(
            engine.db, engine.project_id, node.content, translated_text, engine.locked_glossary,
        )
        if locked_issues:
            signature_value = semantic_signature(
                node.content, translated_text, engine.locked_glossary, entity_context,
                engine.config.semantic_critic_model or engine.config.model_name,
                engine.config.document_type,
            )
            rejected = SemanticAssuranceResult(
                translated_text, False, "FAIL", 1.0, "HIGH",
                [{"type": item["code"], "severity": item["severity"], "message": item["message"]} for item in locked_issues],
                signature_value,
            )
            cls.persist_rejected(engine.db, engine.project_id, node, rejected, model_name)
            return rejected
        result = cls.evaluate_candidate(
            engine, chapter, engine.canonical_node(node), translated_text, entity_context,
            previous_repairs=previous_repairs,
            max_repairs=engine.config.semantic_max_repairs,
        )
        if not result.approved:
            cls.persist_rejected(engine.db, engine.project_id, node, result, model_name)
            return result
        TranslationCommitService.commit_validated_translation(
            engine.db, engine.project_id, node, result.translated_text, signature,
            engine.locked_glossary, model_name, created_by, instruction=instruction,
            resolved_issue_ids=resolved_issue_ids,
            semantic_result=result.audit_payload(engine.config.semantic_critic_model or model_name),
            latency_ms=latency_ms,
        )
        return result

    @classmethod
    def evaluate_candidate(cls, engine, chapter, node, translated_text: str, entity_context: Dict[str, str], previous_repairs: int = 0, max_repairs: int = 2) -> SemanticAssuranceResult:
        risk = SemanticRiskScorer.score(
            node.content, translated_text, getattr(node.type, "value", node.type), engine.config.document_type,
            previous_repairs=previous_repairs, entity_count=len(entity_context),
            medium_threshold=engine.config.semantic_risk_medium,
            high_threshold=engine.config.semantic_risk_high,
        )
        signature = semantic_signature(
            node.content, translated_text, engine.locked_glossary, entity_context,
            engine.config.semantic_critic_model or engine.config.model_name, engine.config.document_type,
        )
        cached = engine.db.query(SemanticReviewModel).filter(
            SemanticReviewModel.project_id == engine.project_id,
            SemanticReviewModel.node_id == node.id,
            SemanticReviewModel.signature == signature,
            SemanticReviewModel.is_stale.is_(False),
        ).first()
        if cached:
            return SemanticAssuranceResult(
                translated_text, cached.critic_status in {"PASS", "NOT_REQUIRED"}, cached.critic_status, cached.risk_score, cached.risk_level,
                list(cached.issues_json or []), signature, cached.critic_score,
                critic_calls=0, from_cache=True,
            )
        if not risk.requires_critic:
            return SemanticAssuranceResult(translated_text, True, "NOT_REQUIRED", risk.score, risk.level, [], signature)

        candidate = translated_text
        total_calls = 0
        total_tokens = 0
        total_latency = 0.0
        for repair_index in range(max_repairs + 1):
            review = SemanticCritic.review(
                engine.provider, node.content, candidate, engine.locked_glossary,
                entity_context, engine.config.document_type, engine.config.semantic_critic_model or engine.config.model_name,
            )
            total_calls += review.critic_calls
            total_tokens += review.critic_request_tokens
            total_latency += review.critic_latency_ms
            current_signature = semantic_signature(
                node.content, candidate, engine.locked_glossary, entity_context,
                engine.config.semantic_critic_model or engine.config.model_name, engine.config.document_type,
            )
            if review.status == "PASS":
                return SemanticAssuranceResult(
                    candidate, True, "PASS", risk.score, risk.level, [], current_signature,
                    review.score, repair_index, total_calls, total_tokens, total_latency,
                )
            if review.status == "ERROR" or repair_index >= max_repairs:
                return SemanticAssuranceResult(
                    candidate, False, review.status, risk.score, risk.level, review.errors, current_signature,
                    review.score, repair_index, total_calls, total_tokens, total_latency,
                )
            repair_issues = [{
                "code": "SEMANTIC_REPAIR_POLICY",
                "message": f"PROMPT VERSION {SEMANTIC_REPAIR_PROMPT_VERSION}: sửa từ ORIGINAL SOURCE; candidate cũ chỉ dùng chẩn đoán.",
            }] + [{"code": item["type"], "message": item["message"]} for item in review.errors]
            repaired = engine.repair_node(chapter, node, repair_issues, max_attempts=1)
            if not repaired.passed:
                return SemanticAssuranceResult(
                    repaired.translated_text, False, "FAIL", risk.score, risk.level,
                    repair_issues + repaired.quality.issues, current_signature,
                    review.score, repair_index + 1, total_calls, total_tokens, total_latency,
                )
            candidate = repaired.translated_text

        raise AssertionError("Vòng semantic repair phải kết thúc có giới hạn")

    @classmethod
    def persist_rejected(cls, db, project_id: str, node, result: SemanticAssuranceResult, model_name: str) -> None:
        """Lưu audit fail-closed và QA issue nhưng không lưu candidate/TM."""
        existing = db.query(SemanticReviewModel).filter(
            SemanticReviewModel.project_id == project_id,
            SemanticReviewModel.node_id == node.id,
            SemanticReviewModel.signature == result.signature,
            SemanticReviewModel.is_stale.is_(False),
        ).first()
        if not existing:
            db.add(SemanticReviewModel(
                id=str(uuid.uuid4()), project_id=project_id, node_id=node.id,
                translation_version=node.version, signature=result.signature,
                risk_score=result.risk_score, risk_level=result.risk_level,
                critic_status=result.status, critic_score=result.score,
                issues_json=result.errors, model_name=model_name,
                prompt_version=SEMANTIC_CRITIC_PROMPT_VERSION,
                critic_request_tokens=result.critic_request_tokens,
                critic_latency_ms=result.critic_latency_ms, critic_calls=result.critic_calls,
            ))
        for error in result.errors:
            issue_type = error.get("type", "SEMANTIC_QA_ERROR")
            open_issue = db.query(QAIssueModel).filter(
                QAIssueModel.project_id == project_id,
                QAIssueModel.node_id == node.id,
                QAIssueModel.issue_type == issue_type,
                QAIssueModel.status == "OPEN",
            ).first()
            if not open_issue:
                db.add(QAIssueModel(
                    id=str(uuid.uuid4()), project_id=project_id, node_id=node.id,
                    issue_type=issue_type, severity=error.get("severity", "ERROR"),
                    message=error.get("message", "Semantic review không đạt."),
                    source_snippet=(node.content or "")[:180],
                    translation_snippet=(result.translated_text or "")[:180], status="OPEN",
                ))
        node.status = "NEEDS_REVIEW"
        db.commit()

    @classmethod
    def persist_existing_review(cls, db, project_id: str, node, result: SemanticAssuranceResult, model_name: str) -> None:
        """Ghi audit cho bản dịch đã tồn tại mà không tạo thêm version/TM."""
        if not result.approved:
            cls.persist_rejected(db, project_id, node, result, model_name)
            return
        review = db.query(SemanticReviewModel).filter(
            SemanticReviewModel.project_id == project_id,
            SemanticReviewModel.node_id == node.id,
            SemanticReviewModel.signature == result.signature,
            SemanticReviewModel.is_stale.is_(False),
        ).first()
        if not review:
            review = SemanticReviewModel(
                id=str(uuid.uuid4()), project_id=project_id, node_id=node.id,
                signature=result.signature,
            )
            db.add(review)
        review.translation_version = node.version
        review.risk_score = result.risk_score
        review.risk_level = result.risk_level
        review.critic_status = result.status
        review.critic_score = result.score
        review.issues_json = result.errors
        review.model_name = model_name
        review.prompt_version = SEMANTIC_CRITIC_PROMPT_VERSION
        review.critic_request_tokens = result.critic_request_tokens
        review.critic_latency_ms = result.critic_latency_ms
        review.critic_calls = result.critic_calls
        db.commit()
