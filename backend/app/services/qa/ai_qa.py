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

            if result.get("status") == "ERROR" or result.get("error"):
                return {
                    "issue_type": "QA_ERROR",
                    "severity": "ERROR",
                    "message": result.get("error") or "Dịch vụ AI QA trả về kết quả không hợp lệ.",
                    "source_snippet": node.content[:150],
                    "translation_snippet": node.translated_content[:150],
                    "suggested_fix": "Kiểm tra provider rồi chạy lại QA. Không coi lỗi này là đạt.",
                }

            if not result.get("is_passed", False):
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
        except Exception as exc:
            return {
                "issue_type": "QA_ERROR",
                "severity": "ERROR",
                "message": f"AI QA thất bại: {exc}",
                "source_snippet": node.content[:150],
                "translation_snippet": node.translated_content[:150],
                "suggested_fix": "Kiểm tra provider rồi chạy lại QA.",
            }

        return None
