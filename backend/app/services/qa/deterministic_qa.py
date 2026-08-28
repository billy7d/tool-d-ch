from typing import List, Dict, Any, Optional
from app.db.models import NodeModel
from app.services.translation.quality_gate import TranslationQualityGate


class DeterministicQA:
    LENGTH_CODES = {"short": "POSSIBLE_TRUNCATION", "long": "POSSIBLE_ADDED_CONTENT"}

    @classmethod
    def audit_node(
        cls,
        node: NodeModel,
        locked_glossary: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Dùng cùng Quality Gate với đường dịch để QA không lệch contract."""
        source = (node.content or "").strip()
        translation = (node.translated_content or "").strip()
        quality = TranslationQualityGate().validate(
            source, translation, locked_glossary or {},
        )
        issues: List[Dict[str, Any]] = []
        for quality_issue in quality.issues:
            code = quality_issue["code"]
            if code == "FOREIGN_SCRIPT_CONTAMINATION":
                code = "CONTAINS_CHINESE"
            if code == "LENGTH_ANOMALY":
                ratio = len(translation) / max(len(source), 1)
                code = cls.LENGTH_CODES["short" if ratio < 0.25 else "long"]
            issues.append({
                "issue_type": code,
                "severity": quality_issue["severity"],
                "message": quality_issue["message"],
                "source_snippet": source[:150],
                "translation_snippet": translation[:150],
                "suggested_fix": "Kiểm tra và sửa candidate theo lỗi Quality Gate.",
            })
        return issues
