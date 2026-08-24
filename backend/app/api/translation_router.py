from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional

from app.db.engine import get_project_db
from app.db.models import NodeModel
from app.db.repository import ProjectRepository, TranslationRepository
from app.models.schemas import TranslationStartRequest, RetranslateNodeRequest, TranslationStatusResponse
from app.models.canonical import TranslationMode
from app.services.translation.worker import translation_worker
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.translation_memory import TranslationMemoryService

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
    
    current_status = "RUNNING" if is_running else ("PAUSED" if is_paused else proj.current_stage)

    return TranslationStatusResponse(
        project_id=project_id,
        status=current_status,
        total_nodes=stats["total_nodes"],
        translated_nodes=stats["translated_nodes"],
        failed_nodes=stats["failed_nodes"],
        needs_review_nodes=stats["needs_review_nodes"],
        progress_percent=stats["progress_percent"],
        current_chapter_title=None,
        current_node_id=None,
        estimated_time_remaining_sec=None,
    )


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
        
        model_name = custom_model or (proj.selected_model if proj else "qwen2.5:7b")
        instruction = custom_instruction or "Dịch lại tự nhiên hơn, chuẩn văn phong xuất bản và đúng ngữ cảnh."

        provider = translation_worker.get_provider(model_name)
        locked_glossary = GlossaryService.get_locked_glossary_map(db, project_id)

        # Build system prompt for high quality
        doc_type = proj.document_type if proj else "GENERAL"
        mode_val = proj.translation_mode if proj else "NATURAL"
        try:
            tr_mode = TranslationMode(mode_val)
        except Exception:
            tr_mode = TranslationMode.NATURAL

        sys_prompt = PromptBuilder.build_system_prompt(
            document_type=doc_type,
            translation_mode=tr_mode,
            style_guide=proj.style_guide if proj else None,
            custom_instructions=instruction
        )

        revised = ""
        # 1. Primary path: Clean single translation directly from English source to prevent Chinese contamination
        try:
            revised = provider.translate_single(
                text=node.content,
                system_prompt=sys_prompt,
                glossary_terms=locked_glossary,
                model=model_name,
                temperature=0.15
            )
        except Exception as e_single:
            print(f"[Retranslate] provider.translate_single failed: {e_single}")

        # 2. Secondary fallback if translate_single produced nothing
        if not revised or not revised.strip():
            try:
                revised = provider.revise(
                    source_text=node.content,
                    current_translation=node.translated_content or "",
                    instruction=instruction,
                    model=model_name
                )
            except Exception as e_rev:
                print(f"[Retranslate] provider.revise failed: {e_rev}")

        # 3. Clean, deduplicate repetition loops, and enforce 100% pure Vietnamese
        if revised:
            revised = VietnamesePostProcessor.clean_vietnamese_text(revised)
            revised = VietnamesePostProcessor.enforce_locked_glossary(revised, node.content, locked_glossary)
            
            # If Chinese character detected, force clean retry at temperature 0.0
            if VietnamesePostProcessor.contains_chinese(revised):
                try:
                    re_clean = provider.translate_single(
                        text=node.content,
                        system_prompt=sys_prompt,
                        glossary_terms=locked_glossary,
                        model=model_name,
                        temperature=0.0
                    )
                    if re_clean:
                        re_clean = VietnamesePostProcessor.clean_vietnamese_text(re_clean)
                        if not VietnamesePostProcessor.contains_chinese(re_clean):
                            revised = re_clean
                except Exception:
                    pass

            # Final guarantee: If any stray Chinese character remains, strip it completely
            if VietnamesePostProcessor.contains_chinese(revised):
                revised = VietnamesePostProcessor.strip_chinese_characters(revised)
                revised = VietnamesePostProcessor.clean_vietnamese_text(revised)

        # Fallback if all attempts yielded empty
        final_translation = revised if (revised and revised.strip()) else (node.translated_content or node.content)

        # 4. Save to Repository and Translation Memory
        trans_repo = TranslationRepository(db)
        trans_repo.save_node_translation(
            node_id=node_id,
            project_id=project_id,
            translated_text=final_translation,
            model_name=model_name,
            instruction=instruction,
            created_by="user_retranslate"
        )
        TranslationMemoryService.store(
            db=db,
            source_text=node.content,
            translated_text=final_translation,
            model_name=model_name
        )

        return {
            "node_id": node_id,
            "translated_content": final_translation,
            "message": "Dịch lại hoàn tất."
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Retranslate Exception] Node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Không thể dịch lại: {str(e)}")
    finally:
        db.close()
