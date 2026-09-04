"""Style Memory theo project, tách biệt với Translation Memory chính xác."""

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.db.models import ChapterModel, NodeModel, StyleMemoryModel
from app.services.translation.entity_ledger import EntityLedgerService
from app.services.translation.few_shot_selector import FewShotSelector
from app.services.translation.quality_gate import QualityGateResult, TranslationQualityGate
from app.services.translation.semantic_assurance import SemanticAssuranceService


STYLE_MEMORY_VERSION = "style-memory-v1-project-scoped"
APPROVED_SOURCES = {"QA_EDITOR", "RETRANSLATION_APPROVED", "MANUAL_FINAL"}


@dataclass(frozen=True)
class StyleMemoryResult:
    ingested: bool
    reason: str = ""
    record_id: Optional[str] = None
    deduplicated: bool = False
    semantic_status: str = ""


def _normalized(text: str) -> str:
    return " ".join(str(text or "").strip().casefold().split())


def _hash(text: str) -> str:
    return hashlib.sha256(_normalized(text).encode("utf-8")).hexdigest()


def _node_type(value: Any) -> str:
    return str(getattr(value, "value", value or "paragraph")).lower()


class StyleMemoryService:
    """Lưu và truy hồi ví dụ đã duyệt bằng scoring deterministic, không dùng vector DB."""

    VERSION = STYLE_MEMORY_VERSION

    @staticmethod
    def source_hash(text: str) -> str:
        return _hash(text)

    @staticmethod
    def approved_hash(text: str) -> str:
        return _hash(text)

    @staticmethod
    def _approved_source(value: str) -> str:
        normalized = str(value or "").strip().upper()
        return normalized if normalized in APPROVED_SOURCES else ""

    @classmethod
    def ingest_approved_example(
        cls,
        db: Session,
        project_id: str,
        source_text: str,
        approved_vi: str,
        *,
        domain: str = "GENERAL",
        document_type: str = "GENERAL",
        register: str = "",
        translation_mode: str = "NATURAL",
        node_type: str = "paragraph",
        source_patterns: Optional[Iterable[str]] = None,
        scope: str = "PROJECT",
        approval_source: str = "QA_EDITOR",
        explicit_approval: bool = False,
        semantic_approved: bool = False,
        semantic_result: Optional[Any] = None,
        quality_result: Optional[QualityGateResult] = None,
        glossary_terms: Optional[Any] = None,
        commit: bool = True,
    ) -> StyleMemoryResult:
        """Chỉ ingest record đã được người dùng xác nhận và qua đủ gate."""
        if not explicit_approval:
            return StyleMemoryResult(False, "EXPLICIT_APPROVAL_REQUIRED")
        if not project_id or not str(source_text or "").strip() or not str(approved_vi or "").strip():
            return StyleMemoryResult(False, "SOURCE_OR_TARGET_EMPTY")
        normalized_source = str(source_text).strip()
        normalized_target = str(approved_vi).strip()
        approved_source = cls._approved_source(approval_source)
        if not approved_source:
            return StyleMemoryResult(False, "UNAPPROVED_SOURCE")
        if str(scope or "PROJECT").upper() not in {"PROJECT", "USER_LOCAL"}:
            return StyleMemoryResult(False, "INVALID_SCOPE")

        gate = quality_result or TranslationQualityGate().validate(
            normalized_source, normalized_target, glossary_terms or {},
        )
        if not gate.passed or any(issue.get("severity") == "ERROR" for issue in gate.issues):
            return StyleMemoryResult(False, "DETERMINISTIC_GATE_FAILED")

        semantic_ok = bool(semantic_approved)
        semantic_status = "PASS" if semantic_ok else ""
        if semantic_result is not None:
            semantic_ok = bool(getattr(semantic_result, "approved", False))
            semantic_status = str(getattr(semantic_result, "status", ""))
        if not semantic_ok:
            return StyleMemoryResult(False, "SEMANTIC_APPROVAL_REQUIRED", semantic_status=semantic_status)

        normalized_domain = str(domain or document_type or "GENERAL").upper()
        normalized_type = _node_type(node_type)
        patterns = sorted(set(source_patterns or FewShotSelector.detect_patterns(normalized_source)))
        source_digest = cls.source_hash(normalized_source)
        target_digest = cls.approved_hash(normalized_target)
        existing = db.query(StyleMemoryModel).filter(
            StyleMemoryModel.project_id == project_id,
            StyleMemoryModel.source_hash == source_digest,
            StyleMemoryModel.approved_hash == target_digest,
            StyleMemoryModel.domain == normalized_domain,
            StyleMemoryModel.translation_mode == str(translation_mode or "NATURAL").upper(),
            StyleMemoryModel.node_type == normalized_type,
            StyleMemoryModel.quality_status == "USER_APPROVED",
        ).first()
        if existing:
            return StyleMemoryResult(False, "DUPLICATE", existing.id, True, semantic_status)

        record = StyleMemoryModel(
            id=str(uuid.uuid4()),
            project_id=project_id,
            source_text=normalized_source,
            approved_vi=normalized_target,
            source_hash=source_digest,
            approved_hash=target_digest,
            domain=normalized_domain,
            document_type=str(document_type or normalized_domain).upper(),
            register=str(register or "").upper(),
            translation_mode=str(translation_mode or "NATURAL").upper(),
            node_type=normalized_type,
            source_patterns=patterns,
            scope=str(scope or "PROJECT").upper(),
            quality_status="USER_APPROVED",
            approval_source=approved_source,
        )
        db.add(record)
        if commit:
            db.commit()
            db.refresh(record)
        else:
            db.flush()
        return StyleMemoryResult(True, "INGESTED", record.id, False, semantic_status)

    @classmethod
    def ingest_approved_node(
        cls,
        db: Session,
        project_id: str,
        node: NodeModel,
        engine: Any,
        chapter: Optional[ChapterModel] = None,
        approval_source: str = "QA_EDITOR",
    ) -> StyleMemoryResult:
        """Đánh giá node hiện tại rồi mới ghi style record; không commit lại candidate/TM."""
        if str(getattr(node, "approval_status", "")).upper() != "APPROVED":
            return StyleMemoryResult(False, "EXPLICIT_APPROVAL_REQUIRED")
        target = str(node.translated_content or "").strip()
        if not target:
            return StyleMemoryResult(False, "SOURCE_OR_TARGET_EMPTY")
        chapter = chapter or db.query(ChapterModel).filter(ChapterModel.id == node.chapter_id).first()
        if not chapter:
            return StyleMemoryResult(False, "CHAPTER_NOT_FOUND")
        glossary = getattr(engine, "glossary_validation_terms", getattr(engine, "locked_glossary", {}))
        quality = TranslationQualityGate().validate(node.content, target, glossary)
        if not quality.passed:
            return StyleMemoryResult(False, "DETERMINISTIC_GATE_FAILED")
        canonical = engine.canonical_node(node)
        entities = EntityLedgerService.relevant_decisions(
            db, project_id, node.content, getattr(engine, "locked_glossary", {}),
        )
        semantic = SemanticAssuranceService.evaluate_candidate(
            engine, chapter, canonical, target, entities, max_repairs=0,
        )
        config = getattr(engine, "config", None)
        return cls.ingest_approved_example(
            db,
            project_id,
            node.content,
            target,
            domain=str(getattr(config, "document_type", "GENERAL") or "GENERAL"),
            document_type=str(getattr(config, "document_type", "GENERAL") or "GENERAL"),
            register=str(getattr(config, "register", "") or ""),
            translation_mode=str(getattr(config, "translation_mode", "NATURAL") or "NATURAL"),
            node_type=_node_type(node.node_type),
            approval_source=approval_source,
            explicit_approval=True,
            semantic_result=semantic,
            quality_result=quality,
            glossary_terms=glossary,
        )

    @classmethod
    def retrieve_examples(
        cls,
        db: Session,
        project_id: str,
        source_text: str,
        *,
        domain: str = "GENERAL",
        document_type: str = "GENERAL",
        register: str = "",
        translation_mode: str = "NATURAL",
        node_type: str = "paragraph",
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Truy hồi 1-3 ví dụ cùng project theo domain/pattern/độ dài."""
        maximum = min(3, max(0, int(limit)))
        if maximum == 0 or not project_id:
            return []
        source_patterns = FewShotSelector.detect_patterns(source_text)
        words = set(re.findall(r"[\wÀ-ỹ]+", _normalized(source_text)))
        normalized_domain = str(domain or "GENERAL").upper()
        normalized_document = str(document_type or "GENERAL").upper()
        normalized_mode = str(translation_mode or "NATURAL").upper()
        normalized_node = _node_type(node_type)
        rows = db.query(StyleMemoryModel).filter(
            StyleMemoryModel.project_id == project_id,
            StyleMemoryModel.quality_status == "USER_APPROVED",
            StyleMemoryModel.scope.in_(["PROJECT", "USER_LOCAL"]),
        ).all()

        def score(row: StyleMemoryModel) -> tuple[int, str]:
            row_patterns = set(row.source_patterns or [])
            row_words = set(re.findall(r"[\wÀ-ỹ]+", _normalized(row.source_text)))
            value = 0
            value += 70 if str(row.domain or "").upper() == normalized_domain else 0
            value += 25 if str(row.document_type or "").upper() == normalized_document else 0
            value += 18 if str(row.translation_mode or "").upper() == normalized_mode else 0
            value += 18 if _node_type(row.node_type) == normalized_node else 0
            value += 12 * len(source_patterns.intersection(row_patterns))
            value += min(20, len(words.intersection(row_words)) * 3)
            value += max(0, 10 - abs(len(words) - len(row_words)) // 4)
            return value, row.id

        selected = sorted(rows, key=lambda row: (-score(row)[0], score(row)[1]))[:maximum]
        return [
            {
                "id": row.id,
                "source_text": row.source_text,
                "approved_vi": row.approved_vi,
                "domain": row.domain,
                "document_type": row.document_type,
                "register": row.register,
                "translation_mode": row.translation_mode,
                "node_type": row.node_type,
                "source_patterns": list(row.source_patterns or []),
                "quality_status": row.quality_status,
            }
            for row in selected
        ]

    @classmethod
    def retrieve(cls, db: Session, project_id: str, source_text: str, **kwargs) -> List[Dict[str, Any]]:
        return cls.retrieve_examples(db, project_id, source_text, **kwargs)

    @classmethod
    def list_project_memory(cls, db: Session, project_id: str) -> List[StyleMemoryModel]:
        return db.query(StyleMemoryModel).filter(
            StyleMemoryModel.project_id == project_id,
            StyleMemoryModel.quality_status == "USER_APPROVED",
        ).order_by(StyleMemoryModel.approved_at.desc(), StyleMemoryModel.id).all()
