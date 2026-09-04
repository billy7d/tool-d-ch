from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from app.db.engine import get_project_db
from app.db.models import NodeModel, ChapterModel, ProjectModel
from app.services.translation.semantic_assurance import SemanticAssuranceService
from app.db.repository import StructureRepository, ProjectRepository
from app.models.canonical import CanonicalDocument, NodeType, DocumentNode
from app.models.canonical import ApprovalStatus
from app.models.schemas import NodeUpdate, StructureConfirmRequest
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.style_memory import StyleMemoryService
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.worker import translation_worker

router = APIRouter(prefix="/api/projects/{project_id}/structure", tags=["Structure"])


@router.get("", response_model=CanonicalDocument)
def get_structure(project_id: str):
    db = get_project_db(project_id)
    repo = StructureRepository(db)
    try:
        doc = repo.get_canonical_document(project_id)
        return doc
    finally:
        db.close()


@router.patch("/nodes/{node_id}")
def update_node(project_id: str, node_id: str, payload: NodeUpdate):
    db = get_project_db(project_id)
    repo = StructureRepository(db)
    try:
        update_dict = {}
        if payload.node_type:
            update_dict["node_type"] = payload.node_type.value
        if payload.content is not None:
            update_dict["content"] = payload.content
        if payload.translated_content is not None:
            update_dict["translated_content"] = payload.translated_content
        if payload.approval_status:
            update_dict["approval_status"] = payload.approval_status.value

        if payload.content is not None or payload.translated_content is not None:
            SemanticAssuranceService.invalidate_for_nodes(db, project_id, [node_id])

        updated = repo.update_node(node_id, **update_dict)
        if not updated:
            raise HTTPException(status_code=404, detail="Không tìm thấy phần tử nội dung (node).")
        style_memory = {"ingested": False, "reason": "NOT_REQUESTED"}
        if payload.approval_status == ApprovalStatus.APPROVED:
            # Approval là hành động rõ ràng của người dùng; chỉ sau đó mới kiểm
            # deterministic + semantic để đưa ví dụ vào Style Memory project.
            try:
                project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
                chapter = db.query(ChapterModel).filter(ChapterModel.id == updated.chapter_id).first()
                if project and chapter and updated.translated_content:
                    config = TranslationConfig.from_project(project)
                    provider = translation_worker.get_provider(config.model_name)
                    glossary = GlossaryService.get_locked_glossary_map(db, project_id)
                    engine = ContextualTranslationEngine(db, project, provider, config, glossary)
                    result = StyleMemoryService.ingest_approved_node(
                        db, project_id, updated, engine, chapter, approval_source="QA_EDITOR",
                    )
                    style_memory = result.__dict__
                else:
                    style_memory = {"ingested": False, "reason": "SOURCE_OR_TARGET_EMPTY"}
            except Exception as exc:
                # Không để lỗi style memory làm mất thao tác duyệt node đã lưu.
                style_memory = {"ingested": False, "reason": "STYLE_MEMORY_VERIFICATION_ERROR", "detail": str(exc)}
        return {"message": "Cập nhật thành công", "node_id": node_id, "style_memory": style_memory}
    finally:
        db.close()


@router.post("/nodes/{node_id}/merge_next")
def merge_with_next_node(project_id: str, node_id: str):
    db = get_project_db(project_id)
    try:
        current_node = db.query(NodeModel).filter(NodeModel.id == node_id).first()
        if not current_node:
            raise HTTPException(status_code=404, detail="Không tìm thấy node.")

        next_node = db.query(NodeModel).filter(
            NodeModel.project_id == project_id,
            NodeModel.chapter_id == current_node.chapter_id,
            NodeModel.order_index > current_node.order_index
        ).order_by(NodeModel.order_index).first()

        if not next_node:
            raise HTTPException(status_code=400, detail="Không có đoạn văn liền sau để gộp.")

        current_node.content = f"{current_node.content} {next_node.content}"
        db.delete(next_node)
        db.commit()
        return {"message": "Gộp đoạn văn thành công."}
    finally:
        db.close()


@router.post("/confirm")
def confirm_structure(project_id: str, payload: StructureConfirmRequest):
    db = get_project_db(project_id)
    proj_repo = ProjectRepository(db)
    try:
        proj = proj_repo.get_project(project_id)
        if not proj:
            raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

        proj_repo.update_project(
            project_id,
            structure_confirmed=True,
            current_stage="STRUCTURE_CONFIRMED",
            structure_version=proj.structure_version + 1
        )
        return {"message": "Đã xác nhận và khóa cấu trúc tài liệu thành công.", "version": proj.structure_version + 1}
    finally:
        db.close()
