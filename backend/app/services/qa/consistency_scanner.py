import re
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.db.models import NodeModel, GlossaryModel


class ConsistencyScanner:
    @staticmethod
    def scan_project(db: Session, project_id: str) -> List[Dict[str, Any]]:
        nodes = db.query(NodeModel).filter(
            NodeModel.project_id == project_id,
            NodeModel.translated_content != None
        ).all()

        glossary_items = db.query(GlossaryModel).filter(
            GlossaryModel.project_id == project_id
        ).all()

        inconsistencies = []

        # Check each glossary term against all translations
        for g in glossary_items:
            src_term = g.source_term.lower()
            expected_target = g.target_term.lower()

            occurrences = []
            deviations = []

            for n in nodes:
                source = (n.content or "").lower()
                translation = (n.translated_content or "").lower()

                if src_term in source:
                    occurrences.append(n.id)
                    # If expected term is absent in translation
                    if expected_target not in translation:
                        deviations.append({
                            "node_id": n.id,
                            "source_snippet": n.content[:150],
                            "translation_snippet": n.translated_content[:150],
                        })

            if deviations and len(occurrences) >= 2:
                inconsistencies.append({
                    "source_term": g.source_term,
                    "expected_target": g.target_term,
                    "total_occurrences": len(occurrences),
                    "deviations_count": len(deviations),
                    "deviations": deviations,
                    "message": f"Thuật ngữ '{g.source_term}' xuất hiện {len(occurrences)} lần nhưng có {len(deviations)} đoạn không sử dụng '{g.target_term}'."
                })

        return inconsistencies
