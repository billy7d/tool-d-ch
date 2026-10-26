from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.engine import get_project_db
from app.services.translation.style_rules import StyleRuleService


router = APIRouter(prefix="/api/projects/{project_id}/style", tags=["Style"])


class StyleRuleStatusUpdate(BaseModel):
    status: str


@router.get("/rules")
def list_style_rules(project_id: str) -> List[Dict[str, Any]]:
    db = get_project_db(project_id)
    try:
        return [StyleRuleService.to_dict(rule) for rule in StyleRuleService.list_rules(db, project_id)]
    finally:
        db.close()


@router.patch("/rules/{rule_id}")
def update_style_rule(project_id: str, rule_id: str, payload: StyleRuleStatusUpdate) -> Dict[str, Any]:
    db = get_project_db(project_id)
    try:
        try:
            rule = StyleRuleService.set_status(db, project_id, rule_id, payload.status)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "INVALID_STYLE_RULE_STATUS", "message": str(exc)})
        if not rule:
            raise HTTPException(status_code=404, detail={"code": "STYLE_RULE_NOT_FOUND", "message": "Không tìm thấy quy tắc."})
        return StyleRuleService.to_dict(rule)
    finally:
        db.close()
