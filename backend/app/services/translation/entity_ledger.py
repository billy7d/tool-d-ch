import hashlib
import json
import re
import uuid
from typing import Dict, List, Tuple

from sqlalchemy.orm import Session

from app.db.models import EntityDecisionModel, QAIssueModel
from app.services.translation.term_matcher import TermMatcher


ENTITY_LEDGER_VERSION = "entity-ledger-v1"
ENTITY_TYPES = {"PERSON", "ORGANIZATION", "PRODUCT", "LOCATION", "ACRONYM", "TECHNICAL_IDENTIFIER", "OTHER"}


class EntityLedgerService:
    CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z0-9.&'-]*(?:\s+[A-Z][A-Za-z0-9.&'-]*){0,4}\b")
    ACRONYM = re.compile(r"\b[A-Z][A-Z0-9-]{1,12}\b")
    GENERIC_ACRONYMS = {"CHAPTER", "SECTION", "TABLE", "FIGURE", "APPENDIX"}

    @classmethod
    def extract_source_entities(cls, source_text: str) -> List[Tuple[str, str]]:
        found: Dict[str, str] = {}
        compact_source = (source_text or "").strip()
        heading_like = len(compact_source.split()) <= 8 and not re.search(r"[.!?]$", compact_source)
        for match in cls.CAPITALIZED.finditer(source_text or ""):
            value = match.group(0).strip()
            value = re.sub(r"^(?:The|A|An)\s+", "", value)
            if value in cls.GENERIC_ACRONYMS:
                continue
            single_plain_title = " " not in value and value[:1].isupper() and value[1:].islower()
            sentence_start = not (source_text or "")[:match.start()].strip() or (source_text or "")[:match.start()].rstrip().endswith((".", "!", "?", ":"))
            if single_plain_title and (sentence_start or heading_like):
                continue
            if len(value) > 2 and value.lower() not in {"the", "this", "that"}:
                found[value] = "OTHER"
        for match in cls.ACRONYM.finditer(source_text or ""):
            if match.group(0) not in cls.GENERIC_ACRONYMS:
                found[match.group(0)] = "ACRONYM"
        return list(found.items())

    @classmethod
    def relevant_decisions(cls, db: Session, project_id: str, source_text: str, locked_glossary: Dict[str, str] | None = None) -> Dict[str, str]:
        glossary = locked_glossary or {}
        decisions = db.query(EntityDecisionModel).filter(EntityDecisionModel.project_id == project_id).all()
        relevant: Dict[str, str] = {}
        for item in decisions:
            if any(item.source_key.casefold() == term.casefold() for term in glossary):
                continue
            if TermMatcher.contains(source_text, item.source_key):
                relevant[item.source_key] = item.preferred_translation
        return relevant

    @classmethod
    def validate_locked(cls, db: Session, project_id: str, source_text: str, translated_text: str, locked_glossary: Dict[str, str] | None = None) -> List[Dict[str, str]]:
        decisions = db.query(EntityDecisionModel).filter(
            EntityDecisionModel.project_id == project_id,
            EntityDecisionModel.locked.is_(True),
        ).all()
        glossary = locked_glossary or {}
        issues = []
        for item in decisions:
            glossary_target = next((target for source, target in glossary.items() if source.casefold() == item.source_key.casefold()), None)
            expected = glossary_target or item.preferred_translation
            if TermMatcher.contains(source_text, item.source_key) and not TermMatcher.contains(translated_text, expected):
                issues.append({
                    "code": "ENTITY_MISMATCH", "severity": "ERROR",
                    "message": f'Entity bị khóa "{item.source_key}" phải dịch là "{expected}".',
                })
        return issues

    @classmethod
    def observe_validated(cls, db: Session, project_id: str, node_id: str, source_text: str, translated_text: str) -> None:
        for source_key, entity_type in cls.extract_source_entities(source_text):
            existing = db.query(EntityDecisionModel).filter(
                EntityDecisionModel.project_id == project_id,
                EntityDecisionModel.source_key == source_key,
            ).first()
            # Chỉ tự suy ra khi entity được giữ nguyên; không đoán cặp dịch không có alignment.
            observed = source_key if TermMatcher.contains(translated_text, source_key) else None
            if not existing and observed:
                db.add(EntityDecisionModel(
                    id=str(uuid.uuid4()), project_id=project_id, source_key=source_key,
                    entity_type=entity_type, preferred_translation=observed, source="INFERRED",
                    first_node_id=node_id, confidence=0.55, occurrences=1,
                ))
            elif existing:
                existing.occurrences = (existing.occurrences or 0) + 1
                if observed and observed.casefold() != existing.preferred_translation.casefold():
                    existing.conflicts = (existing.conflicts or 0) + 1
                    db.add(QAIssueModel(
                        id=str(uuid.uuid4()), project_id=project_id, node_id=node_id,
                        issue_type="ENTITY_VARIANT", severity="WARNING",
                        message=f'Entity "{source_key}" không theo quyết định "{existing.preferred_translation}".',
                        source_snippet=source_text[:180], translation_snippet=translated_text[:180], status="OPEN",
                    ))

    @classmethod
    def ledger_hash(cls, decisions: Dict[str, str]) -> str:
        return hashlib.sha256(json.dumps(decisions, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
