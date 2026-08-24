from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.db.models import NodeModel
from app.db.repository import QARepository
from app.services.translation.worker import translation_worker
from app.services.translation.glossary_service import GlossaryService


class AIQAEngine:
    @staticmethod
    def audit_node_ai(db: Session, project_id: str, node: NodeModel, model_name: str = "qwen2.5:7b") -> Optional[Dict[str, Any]]:
        if not node.translated_content or not node.content:
            return None

        provider = translation_worker.get_provider(model_name)
        glossary_map = GlossaryService.get_locked_glossary_map(db, project_id)

        try:
            result = provider.review_translation(
                source_text=node.content,
                translated_text=node.translated_content,
                glossary_terms=glossary_map,
                model=model_name
            )

            if not result.get("is_passed", True):
                issues = result.get("issues", [])
                msg = "; ".join(issues) if issues else "Bản dịch có thể chưa hoàn toàn chuẩn xác hoặc tự nhiên."
                return {
                    "issue_type": "AI_QA_RECOMMENDATION",
                    "severity": "INFO",
                    "message": msg,
                    "source_snippet": node.content[:150],
                    "translation_snippet": node.translated_content[:150],
                    "suggested_fix": result.get("suggested_revision")
                }
        except Exception as e:
            print(f"[AI QA] Error evaluating node {node.id}: {e}")

        return None
