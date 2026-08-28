from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.db.engine import get_project_db
from app.db.models import EntityDecisionModel
from app.services.translation.entity_ledger import ENTITY_TYPES
from app.services.translation.semantic_assurance import SemanticAssuranceService


router = APIRouter(prefix="/api/projects/{project_id}/entities", tags=["Entities"])


@router.get("")
def list_entities(project_id: str):
    db = get_project_db(project_id)
    try:
        rows = db.query(EntityDecisionModel).filter(
            EntityDecisionModel.project_id == project_id,
        ).order_by(EntityDecisionModel.source_key).all()
        return [{
            "id": row.id, "source_key": row.source_key,
            "preferred_translation": row.preferred_translation,
            "entity_type": row.entity_type, "aliases": row.aliases_json or [],
            "locked": row.locked, "confidence": row.confidence,
            "source": row.source, "occurrences": row.occurrences,
            "conflicts": row.conflicts, "revision": row.revision,
        } for row in rows]
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
