import uuid
from typing import Any, Dict, Iterable, Optional

from sqlalchemy.orm import Session

from app.db.models import QAIssueModel, SemanticReviewModel
from app.db.repository import TranslationRepository
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.translation_memory import TranslationMemoryService
from app.services.translation.entity_ledger import EntityLedgerService


class TranslationCommitService:
    @classmethod
    def commit_validated_translation(
        cls,
        db: Session,
        project_id: str,
        node: Any,
        translated_text: str,
        signature: Any,
        locked_glossary: Dict[str, str],
        model_name: str,
        created_by: str,
        instruction: Optional[str] = None,
        resolved_issue_ids: Optional[Iterable[str]] = None,
        semantic_result: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
    ):
        """Lưu node/version/TM/QA/semantic trong một transaction duy nhất."""
        quality = TranslationQualityGate().validate(node.content, translated_text, locked_glossary or {})
        if not quality.passed:
            codes = ", ".join(issue["code"] for issue in quality.issues if issue["severity"] == "ERROR")
            raise ValueError(f"Candidate không qua Quality Gate cuối: {codes}")
        if semantic_result and semantic_result.get("status") not in {"PASS", "NOT_REQUIRED"}:
            raise ValueError("Candidate chưa được semantic policy cho phép lưu")
        entity_issues = EntityLedgerService.validate_locked(
            db, project_id, node.content, translated_text, locked_glossary,
        )
        if entity_issues:
            raise ValueError(entity_issues[0]["message"])
        try:
            # TM được stage trước để failure ở node/version vẫn rollback toàn bộ.
            TranslationMemoryService.store(
                db=db,
                source_text=node.content,
                translated_text=translated_text,
                style_hash=signature.style_hash,
                glossary_hash=signature.glossary_hash,
                model_name=model_name,
                prompt_version=signature.prompt_version,
                locked_glossary=locked_glossary,
                commit=False,
            )
            translation = TranslationRepository(db).save_node_translation(
                node_id=node.id,
                project_id=project_id,
                translated_text=translated_text,
                model_name=model_name,
                instruction=instruction,
                created_by=created_by,
                prompt_version=signature.prompt_version,
                latency_ms=latency_ms,
                commit=False,
            )
            if resolved_issue_ids:
                db.query(QAIssueModel).filter(
                    QAIssueModel.project_id == project_id,
                    QAIssueModel.id.in_(list(resolved_issue_ids)),
                ).update({"status": "RESOLVED"}, synchronize_session="fetch")
            if semantic_result:
                review = db.query(SemanticReviewModel).filter(
                    SemanticReviewModel.project_id == project_id,
                    SemanticReviewModel.node_id == node.id,
                    SemanticReviewModel.signature == semantic_result["signature"],
                    SemanticReviewModel.is_stale.is_(False),
                ).first()
                if not review:
                    review = SemanticReviewModel(
                        id=str(uuid.uuid4()), project_id=project_id, node_id=node.id,
                        signature=semantic_result["signature"],
                    )
                    db.add(review)
                review.translation_version = node.version
                review.risk_score = semantic_result.get("risk_score", 0.0)
                review.risk_level = semantic_result.get("risk_level", "LOW")
                review.critic_status = semantic_result.get("status", "NOT_REQUIRED")
                review.critic_score = semantic_result.get("score")
                review.issues_json = semantic_result.get("errors", [])
                review.model_name = semantic_result.get("model_name", model_name)
                review.prompt_version = semantic_result.get("prompt_version", "semantic-critic-v1")
                review.critic_request_tokens = semantic_result.get("critic_request_tokens", 0)
                review.critic_latency_ms = semantic_result.get("critic_latency_ms", 0.0)
                review.critic_calls = semantic_result.get("critic_calls", 0)
                db.flush()
            EntityLedgerService.observe_validated(
                db, project_id, node.id, node.content, translated_text,
            )
            db.flush()
            db.commit()
            return translation
        except Exception:
            db.rollback()
            raise
