from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List

from app.db.engine import get_project_db
from app.db.models import NodeModel, ChapterModel
from app.db.repository import ProjectRepository
from app.models.schemas import (
    TranslationStartRequest, RetranslateNodeRequest, TranslationStatusResponse,
    TranslationPreviewRequest, TranslationPreviewResponse,
)
from app.services.translation.worker import translation_worker
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.translation_memory import TranslationMemoryService
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.node_policy import is_translatable_node_type
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.translation_signature import (
    PROMPT_VERSION,
    build_translation_signature_from_config,
)
from app.services.translation.semantic_assurance import SemanticAssuranceService

router = APIRouter(prefix="/api/projects/{project_id}/translation", tags=["Translation"])


@router.post("/start")
def start_translation(project_id: str, payload: TranslationStartRequest):
    db = get_project_db(project_id)
    proj_repo = ProjectRepository(db)
    proj = proj_repo.get_project(project_id)
    if not proj:
        db.close()
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    model_to_use = payload.model_name or proj.selected_model or "qwen2.5:7b"
    try:
        PromptBuilder.validate_custom_instructions(payload.custom_instructions or proj.custom_instructions)
    except ValueError as exc:
        db.close()
        raise HTTPException(status_code=400, detail=str(exc))
    if payload.translation_mode:
        proj_repo.update_project(project_id, translation_mode=payload.translation_mode.value)
    if payload.custom_instructions:
        proj_repo.update_project(project_id, custom_instructions=payload.custom_instructions)
    db.close()

    translation_worker.start_translation(
        project_id=project_id,
        model_name=model_to_use,
        custom_instructions=payload.custom_instructions
    )
    return {"message": "Đã bắt đầu tiến trình dịch."}


@router.post("/pause")
def pause_translation(project_id: str):
    translation_worker.pause_translation(project_id)
    return {"message": "Đã tạm dừng tiến trình dịch."}


@router.post("/resume")
def resume_translation(project_id: str):
    translation_worker.resume_translation(project_id)
    return {"message": "Đã tiếp tục tiến trình dịch."}


@router.post("/stop")
def stop_translation(project_id: str):
    translation_worker.stop_translation(project_id)
    return {"message": "Đã dừng tiến trình dịch."}


@router.post("/retry_failed")
def retry_failed_nodes(project_id: str):
    translation_worker.retry_failed(project_id)
    return {"message": "Đang thử dịch lại các đoạn văn bản bị lỗi."}


@router.get("/status", response_model=TranslationStatusResponse)
def get_translation_status(project_id: str):
    db = get_project_db(project_id)
    proj_repo = ProjectRepository(db)
    stats = proj_repo.get_project_stats(project_id)
    proj = proj_repo.get_project(project_id)
    db.close()

    if not proj:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    is_running = project_id in translation_worker.running_tasks and translation_worker.running_tasks[project_id].is_set()
    is_paused = project_id in translation_worker.pause_tasks and translation_worker.pause_tasks[project_id].is_set()
    
    telemetry = translation_worker.telemetry.get(project_id, {})
    execution_status = "PAUSED" if is_paused else (
        "RUNNING" if is_running else telemetry.get("execution_status", "IDLE")
    )
    current_status = execution_status if execution_status != "IDLE" else proj.current_stage
    return TranslationStatusResponse(
        project_id=project_id,
        status=current_status,
        total_nodes=stats["total_nodes"],
        translatable_nodes=stats["translatable_nodes"],
        skipped_nodes=stats["skipped_nodes"],
        translated_nodes=stats["translated_nodes"],
        failed_nodes=stats["failed_nodes"],
        needs_review_nodes=stats["needs_review_nodes"],
        progress_percent=stats["progress_percent"],
        current_chapter_title=telemetry.get("current_chapter_title"),
        current_node_id=None,
        estimated_time_remaining_sec=None,
        current_chunk_id=telemetry.get("current_chunk_id"),
        context_mode=telemetry.get("context_mode", "CONTEXTUAL_BALANCED"),
        retry_count=telemetry.get("retry_count", 0),
        quality_state=telemetry.get("quality_state", "READY"),
        execution_status=execution_status,
        document_status=proj.current_stage,
    )


def _select_preview_nodes(nodes: List[NodeModel], glossary: Dict[str, str], limit: int = 7) -> List[NodeModel]:
    usable = [node for node in nodes if node.content.strip() and is_translatable_node_type(node.node_type)]
    selected: List[NodeModel] = []
    def add(node: Optional[NodeModel]) -> None:
        if node and node.id not in {item.id for item in selected} and len(selected) < limit:
            selected.append(node)
    add(next((node for node in usable if (node.node_type or "").lower() == "heading"), None))
    paragraphs = [node for node in usable if (node.node_type or "").lower() == "paragraph"]
    add(paragraphs[0] if paragraphs else None)
    add(paragraphs[len(paragraphs) // 2] if paragraphs else None)
    add(max(paragraphs, key=lambda item: len(item.content), default=None))
    add(next((node for node in usable if any(term.lower() in node.content.lower() for term in glossary)), None))
    import re
    add(next((node for node in usable if len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", node.content)) >= 1), None))
    for node in usable:
        add(node)
    return selected


@router.post("/preview", response_model=TranslationPreviewResponse)
def preview_translation(project_id: str, payload: TranslationPreviewRequest):
    db = get_project_db(project_id)
    try:
        project = ProjectRepository(db).get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
        try:
            PromptBuilder.validate_custom_instructions(payload.custom_instructions)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        config = TranslationConfig.from_project(
            project,
            model_override=payload.model_name,
            custom_instruction_override=payload.custom_instructions,
            translation_mode_override=payload.translation_mode.value,
            document_type_override=payload.document_type.value,
            register_override=payload.style_register,
            sentence_style_override=payload.sentence_style,
        )
        provider = translation_worker.get_provider(config.model_name)
        if not provider.health_check():
            raise HTTPException(status_code=503, detail="Mô hình local chưa sẵn sàng.")
        all_nodes = db.query(NodeModel).filter(NodeModel.project_id == project_id).order_by(NodeModel.order_index).all()
        glossary = GlossaryService.get_locked_glossary_map(db, project_id)
        selected = _select_preview_nodes(all_nodes, glossary)
        engine = ContextualTranslationEngine(db, project, provider, config, glossary)
        samples = []
        for node in selected:
            chapter = db.query(ChapterModel).filter(ChapterModel.id == node.chapter_id).first()
            if not chapter:
                continue
            result = engine.preview_node(chapter, engine.canonical_node(node))
            samples.append({
                "node_id": node.id,
                "source": node.content,
                "translation": result.translated_text,
                "quality": {"passed": result.passed, "issues": result.quality.issues},
            })
        return {"samples": samples, "profile": engine.document_profile.to_dict(), "prompt_version": PROMPT_VERSION}
    finally:
        db.close()


@router.post("/nodes/{node_id}/retranslate")
def retranslate_single_node(project_id: str, node_id: str, payload: Optional[RetranslateNodeRequest] = None):
    db = get_project_db(project_id)
    try:
        node = db.query(NodeModel).filter(NodeModel.id == node_id).first()
        if not node:
            raise HTTPException(status_code=404, detail="Không tìm thấy node.")

        proj_repo = ProjectRepository(db)
        proj = proj_repo.get_project(project_id)
        
        custom_model = payload.custom_model if payload else None
        custom_instruction = payload.instruction if payload else None
        
        instruction = custom_instruction or "Dịch lại tự nhiên hơn, chuẩn văn phong xuất bản và đúng ngữ cảnh."
        try:
            PromptBuilder.validate_custom_instructions(instruction)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        config = TranslationConfig.from_project(
            proj,
            model_override=custom_model,
            custom_instruction_override=instruction,
        )
        provider = translation_worker.get_provider(config.model_name)
        locked_glossary = GlossaryService.get_locked_glossary_map(db, project_id)
        chapter = db.query(ChapterModel).filter(ChapterModel.id == node.chapter_id).first()
        if not chapter:
            raise HTTPException(status_code=422, detail="Node không thuộc chương hợp lệ.")
        engine = ContextualTranslationEngine(db, proj, provider, config, locked_glossary)
        result = engine.repair_node(
            chapter,
            engine.canonical_node(node),
            [{"code": "USER_RETRANSLATE", "message": instruction}],
        )
        if not result.passed:
            node.status = "NEEDS_REVIEW"
            db.commit()
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "TRANSLATION_VALIDATION_FAILED",
                    "issues": result.quality.issues,
                },
            )

        signature = build_translation_signature_from_config(config, locked_glossary)
        semantic = SemanticAssuranceService.assure_and_commit(
            engine, chapter, node, result.translated_text, signature, config.model_name,
            "user_retranslate", instruction=instruction,
            previous_repairs=max(1, result.attempts - 1),
        )
        if not semantic.approved:
            raise HTTPException(status_code=422, detail={
                "code": "SEMANTIC_VALIDATION_FAILED", "status": semantic.status,
                "issues": semantic.errors,
            })

        return {
            "node_id": node_id,
            "translated_content": semantic.translated_text,
            "message": "Dịch lại hoàn tất."
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Retranslate Exception] Node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Không thể dịch lại: {str(e)}")
    finally:
        db.close()
