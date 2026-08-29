import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import IntegrityError

from app.db.engine import get_project_db
from app.db.models import EntityDecisionModel, GlossaryModel, ProjectModel
from app.models.schemas import EntityDecisionCreate
from app.services.translation.entity_ledger import ENTITY_TYPES
from app.services.translation.semantic_assurance import SemanticAssuranceService


router = APIRouter(prefix="/api/projects/{project_id}/entities", tags=["Entities"])


def _serialize_entity(row: EntityDecisionModel) -> Dict[str, Any]:
    return {
        "id": row.id, "source_key": row.source_key,
        "preferred_translation": row.preferred_translation,
        "entity_type": row.entity_type, "aliases": row.aliases_json or [],
        "locked": row.locked, "confidence": row.confidence,
        "source": row.source, "occurrences": row.occurrences,
        "conflicts": row.conflicts, "revision": row.revision,
    }


@router.get("")
def list_entities(project_id: str):
    db = get_project_db(project_id)
    try:
        rows = db.query(EntityDecisionModel).filter(
            EntityDecisionModel.project_id == project_id,
        ).order_by(EntityDecisionModel.source_key).all()
        return [_serialize_entity(row) for row in rows]
    finally:
        db.close()


@router.post("", status_code=201)
def create_entity(project_id: str, payload: EntityDecisionCreate):
    db = get_project_db(project_id)
    try:
        if not db.query(ProjectModel).filter(ProjectModel.id == project_id).first():
            raise HTTPException(status_code=404, detail={"code": "PROJECT_NOT_FOUND", "message": "Không tìm thấy dự án."})

        source_key = payload.source_key.strip()
        preferred_translation = payload.preferred_translation.strip()
        entity_type = payload.entity_type.strip().upper()
        if not source_key or not preferred_translation:
            raise HTTPException(status_code=422, detail={"code": "INVALID_ENTITY", "message": "Source và bản dịch entity không được rỗng."})
        if entity_type not in ENTITY_TYPES:
            raise HTTPException(status_code=422, detail={"code": "INVALID_ENTITY_TYPE", "message": "Loại entity không hợp lệ."})

        existing = next((item for item in db.query(EntityDecisionModel).filter(
            EntityDecisionModel.project_id == project_id,
        ).all() if item.source_key.casefold() == source_key.casefold()), None)
        if existing:
            raise HTTPException(status_code=409, detail={"code": "ENTITY_ALREADY_EXISTS", "message": "Entity này đã tồn tại; hãy dùng PATCH để chỉnh sửa."})

        glossary = next((item for item in db.query(GlossaryModel).filter(
            GlossaryModel.project_id == project_id,
            GlossaryModel.locked.is_(True),
        ).all() if item.source_term.casefold() == source_key.casefold()), None)
        if glossary and glossary.target_term.strip().casefold() != preferred_translation.casefold():
            raise HTTPException(status_code=409, detail={
                "code": "ENTITY_GLOSSARY_CONFLICT",
                "message": f'Entity phải theo thuật ngữ khóa "{glossary.target_term}".',
            })

        aliases: list[str] = []
        for alias in payload.aliases:
            value = alias.strip()
            if value and value.casefold() != source_key.casefold() and value.casefold() not in {item.casefold() for item in aliases}:
                aliases.append(value)
        row = EntityDecisionModel(
            id=str(uuid.uuid4()), project_id=project_id, source_key=source_key,
            preferred_translation=preferred_translation, entity_type=entity_type,
            aliases_json=aliases, locked=payload.locked, confidence=1.0,
            source="USER", revision=1, occurrences=1, conflicts=0,
        )
        db.add(row)
        # Tạo entity thủ công phải làm stale review bị ảnh hưởng trước khi commit.
        SemanticAssuranceService.invalidate_for_source_term(db, project_id, source_key)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=409, detail={"code": "ENTITY_ALREADY_EXISTS", "message": "Entity này đã tồn tại; hãy dùng PATCH để chỉnh sửa."})
        db.refresh(row)
        return _serialize_entity(row)
    finally:
        db.close()


@router.patch("/{entity_id}")
def update_entity(project_id: str, entity_id: str, payload: Dict[str, Any]):
    db = get_project_db(project_id)
    try:
        row = db.query(EntityDecisionModel).filter(
            EntityDecisionModel.project_id == project_id,
            EntityDecisionModel.id == entity_id,
        ).first()
        if not row:
            raise HTTPException(status_code=404, detail="Không tìm thấy entity.")
        if "preferred_translation" in payload:
            value = str(payload["preferred_translation"]).strip()
            if not value:
                raise HTTPException(status_code=422, detail="Bản dịch entity không được rỗng.")
            row.preferred_translation = value
        if "locked" in payload:
            row.locked = bool(payload["locked"])
        if "entity_type" in payload:
            value = str(payload["entity_type"]).upper()
            if value not in ENTITY_TYPES:
                raise HTTPException(status_code=422, detail="Loại entity không hợp lệ.")
            row.entity_type = value
        row.revision = (row.revision or 0) + 1
        SemanticAssuranceService.invalidate_for_source_term(db, project_id, row.source_key)
        db.commit()
        return {
            "id": row.id, "preferred_translation": row.preferred_translation,
            "locked": row.locked, "entity_type": row.entity_type,
            "revision": row.revision,
        }
    finally:
        db.close()
