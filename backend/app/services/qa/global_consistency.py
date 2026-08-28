import uuid
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from app.db.models import EntityDecisionModel, GlossaryModel, NodeModel, QAIssueModel
from app.services.translation.term_matcher import TermMatcher


GLOBAL_CONSISTENCY_VERSION = "global-consistency-v1"


class GlobalConsistencyScanner:
    @classmethod
    def scan_project(cls, db: Session, project_id: str, persist: bool = True) -> List[Dict[str, Any]]:
        nodes = db.query(NodeModel).filter(
            NodeModel.project_id == project_id,
            NodeModel.translated_content.isnot(None),
        ).order_by(NodeModel.order_index).all()
        decisions = db.query(EntityDecisionModel).filter(EntityDecisionModel.project_id == project_id).all()
        findings: List[Dict[str, Any]] = []
        for decision in decisions:
            occurrences = [node for node in nodes if TermMatcher.contains(node.content or "", decision.source_key)]
            if len(occurrences) < 2:
                continue
            variants: Dict[str, List[str]] = {}
            known = [decision.preferred_translation, *(decision.aliases_json or [])]
            for node in occurrences:
                matched = next((value for value in known if value and TermMatcher.contains(node.translated_content or "", value)), None)
                label = matched or "<KHÔNG KHỚP QUYẾT ĐỊNH>"
                variants.setdefault(label, []).append(node.id)
            if len(variants) <= 1:
                continue
            issue_type = "ENTITY_MISMATCH" if decision.locked else (
                "ACRONYM_INCONSISTENCY" if decision.entity_type == "ACRONYM" else "ENTITY_VARIANT"
            )
            severity = "ERROR" if decision.locked else "WARNING"
            finding = {
                "issue_type": issue_type, "severity": severity,
                "source_key": decision.source_key,
                "preferred_translation": decision.preferred_translation,
                "occurrences": len(occurrences), "variants": variants,
                "message": f'Entity "{decision.source_key}" có {len(variants)} cách thể hiện trong {len(occurrences)} node.',
            }
            findings.append(finding)
            if persist:
                existing = db.query(QAIssueModel).filter(
                    QAIssueModel.project_id == project_id,
                    QAIssueModel.node_id.is_(None),
                    QAIssueModel.issue_type == issue_type,
                    QAIssueModel.message == finding["message"],
                    QAIssueModel.status == "OPEN",
                ).first()
                if not existing:
                    db.add(QAIssueModel(
                        id=str(uuid.uuid4()), project_id=project_id, node_id=None,
                        issue_type=issue_type, severity=severity, message=finding["message"], status="OPEN",
                    ))
        glossary_items = db.query(GlossaryModel).filter(GlossaryModel.project_id == project_id).all()
        for term in glossary_items:
            occurrences = [node for node in nodes if TermMatcher.contains(node.content or "", term.source_term)]
            deviations = [node for node in occurrences if not TermMatcher.contains(node.translated_content or "", term.target_term)]
            if len(occurrences) >= 2 and deviations:
                finding = {
                    "issue_type": "ENTITY_MISMATCH" if term.locked else "TERM_VARIANT",
                    "severity": "ERROR" if term.locked else "WARNING",
                    "source_key": term.source_term, "preferred_translation": term.target_term,
                    "occurrences": len(occurrences), "variants": {"deviation_node_ids": [node.id for node in deviations]},
                    "message": f'Thuật ngữ "{term.source_term}" có {len(deviations)}/{len(occurrences)} node không dùng "{term.target_term}".',
                }
                findings.append(finding)
        repeated: Dict[str, List[NodeModel]] = {}
        for node in nodes:
            key = " ".join((node.content or "").casefold().split())
            if len(key.split()) >= 3:
                repeated.setdefault(key, []).append(node)
        for source_key, group in repeated.items():
            variants = {" ".join((node.translated_content or "").casefold().split()) for node in group}
            if len(group) >= 2 and len(variants) > 1:
                findings.append({
                    "issue_type": "REPEATED_PHRASE_INCONSISTENCY", "severity": "WARNING",
                    "source_key": source_key[:160], "preferred_translation": "",
                    "occurrences": len(group), "variants": {value: [] for value in sorted(variants)},
                    "message": f"Cùng một cụm nguồn được dịch theo {len(variants)} cách.",
                })
        if persist:
            for finding in findings:
                existing = db.query(QAIssueModel).filter(
                    QAIssueModel.project_id == project_id,
                    QAIssueModel.node_id.is_(None),
                    QAIssueModel.issue_type == finding["issue_type"],
                    QAIssueModel.message == finding["message"],
                    QAIssueModel.status == "OPEN",
                ).first()
                if not existing:
                    db.add(QAIssueModel(
                        id=str(uuid.uuid4()), project_id=project_id, node_id=None,
                        issue_type=finding["issue_type"], severity=finding["severity"],
                        message=finding["message"], status="OPEN",
                    ))
            db.commit()
        return findings
