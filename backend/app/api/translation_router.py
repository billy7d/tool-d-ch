from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List

from app.db.engine import get_project_db
from app.db.models import NodeModel, ChapterModel
from app.db.repository import ProjectRepository, TranslationRepository
from app.models.schemas import (
    TranslationStartRequest, RetranslateNodeRequest, TranslationStatusResponse,
    TranslationPreviewRequest, TranslationPreviewResponse,
)
from app.models.canonical import TranslationMode, DocumentNode, NodeType, NodeStatus
from app.services.translation.worker import translation_worker
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.translation_memory import TranslationMemoryService
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.translation_signature import build_translation_signature
from app.services.translation.document_profiler import DocumentProfiler
from app.services.translation.context_memory import ChapterMemoryBuilder, RollingContextService
from app.services.translation.context_assembler import ContextAssembler
from app.services.translation.prompt_profiles import select_few_shots
from app.services.translation.prompt_builder import PROMPT_VERSION
from app.config import settings

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
    
    current_status = "RUNNING" if is_running else ("PAUSED" if is_paused else proj.current_stage)

    telemetry = translation_worker.telemetry.get(project_id, {})
    return TranslationStatusResponse(
        project_id=project_id,
        status=current_status,
        total_nodes=stats["total_nodes"],
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
    )


def _canonical_node(node: NodeModel) -> DocumentNode:
    try:
        node_type = NodeType((node.node_type or "paragraph").lower())
    except ValueError:
        node_type = NodeType.PARAGRAPH
    try:
        status = NodeStatus(node.status)
    except ValueError:
        status = NodeStatus.PENDING
    return DocumentNode(id=node.id, type=node_type, content=node.content, status=status, order_index=node.order_index)


def _select_preview_nodes(nodes: List[NodeModel], glossary: Dict[str, str], limit: int = 7) -> List[NodeModel]:
    usable = [node for node in nodes if node.content.strip() and (node.node_type or "").lower() != "code_block"]
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
        model_name = payload.model_name or project.selected_model
        provider = translation_worker.get_provider(model_name)
        if not provider.health_check():
            raise HTTPException(status_code=503, detail="Mô hình local chưa sẵn sàng.")
        all_nodes = db.query(NodeModel).filter(NodeModel.project_id == project_id).order_by(NodeModel.order_index).all()
        glossary = GlossaryService.get_locked_glossary_map(db, project_id)
        selected = _select_preview_nodes(all_nodes, glossary)
        cache_dir = settings.PROJECTS_DIR / project_id / "cache"
        profile = DocumentProfiler.load_or_create(
            all_nodes, cache_dir, payload.document_type.value,
            payload.custom_instructions or "", provider, model_name,
        )
        system_prompt = PromptBuilder.build_system_prompt(
            payload.document_type.value, payload.translation_mode,
            project.style_guide or {}, payload.custom_instructions,
            payload.style_register, payload.sentence_style,
        )
        samples = []
        capabilities = provider.get_model_capabilities(model_name)
        for node in selected:
            chapter = db.query(ChapterModel).filter(ChapterModel.id == node.chapter_id).first()
            chapter_nodes = db.query(NodeModel).filter(NodeModel.chapter_id == node.chapter_id).order_by(NodeModel.order_index).all()
            memory = ChapterMemoryBuilder.load_or_create(
                node.chapter_id or "root", chapter.title if chapter else "Tài liệu",
                chapter_nodes, cache_dir, glossary, provider, model_name,
            )
            canonical = _canonical_node(node)
            rolling = RollingContextService.get_neighbors(db, node, 2, 1, 600)
            context = ContextAssembler.assemble_context(
                [canonical], profile, memory, rolling, glossary, capabilities,
                select_few_shots(payload.document_type.value, payload.translation_mode.value, canonical.type.value),
            )
            user_prompt = PromptBuilder.build_user_prompt(
                [canonical], chapter.title if chapter else "", translation_context=context,
                document_type=payload.document_type.value, translation_mode=payload.translation_mode,
            )
            try:
                result = provider.translate(
                    [{"id": node.id, "text": node.content}], system_prompt, user_prompt,
                    model=model_name, temperature=0.2,
                )
                candidate = next((item.get("text", "") for item in result if item.get("node_id") == node.id), "")
            except Exception as exc:
                candidate = ""
            candidate = VietnamesePostProcessor.normalize_safely(candidate)
            quality = TranslationQualityGate().validate(node.content, candidate, context.glossary)
            samples.append({
                "node_id": node.id,
                "source": node.content,
                "translation": candidate,
                "quality": {"passed": quality.passed, "issues": quality.issues},
            })
        return {"samples": samples, "profile": profile.to_dict(), "prompt_version": PROMPT_VERSION}
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
        
        model_name = custom_model or (proj.selected_model if proj else "qwen2.5:7b")
        instruction = custom_instruction or "Dịch lại tự nhiên hơn, chuẩn văn phong xuất bản và đúng ngữ cảnh."
        try:
            PromptBuilder.validate_custom_instructions(instruction)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

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

        all_nodes = db.query(NodeModel).filter(NodeModel.project_id == project_id).order_by(NodeModel.order_index).all()
        chapter = db.query(ChapterModel).filter(ChapterModel.id == node.chapter_id).first()
        chapter_nodes = db.query(NodeModel).filter(NodeModel.chapter_id == node.chapter_id).order_by(NodeModel.order_index).all()
        cache_dir = settings.PROJECTS_DIR / project_id / "cache"
        profile = DocumentProfiler.load_or_create(
            all_nodes, cache_dir, doc_type, proj.custom_instructions or "", provider, model_name,
        )
        chapter_memory = ChapterMemoryBuilder.load_or_create(
            node.chapter_id or "root", chapter.title if chapter else "Tài liệu",
            chapter_nodes, cache_dir, locked_glossary, provider, model_name,
        )
        canonical = _canonical_node(node)
        neighbors = RollingContextService.get_neighbors(db, node)
        context = ContextAssembler.assemble_context(
            [canonical], profile, chapter_memory, neighbors, locked_glossary,
            provider.get_model_capabilities(model_name),
            select_few_shots(doc_type, tr_mode.value, canonical.type.value),
        )
        contextual_user_prompt = PromptBuilder.build_user_prompt(
            [canonical], chapter.title if chapter else "", translation_context=context,
            document_type=doc_type, translation_mode=tr_mode,
        )

        signature = build_translation_signature(
            source_language=proj.source_language or "en",
            target_language=proj.target_language or "vi",
            translation_mode=tr_mode.value,
            document_type=doc_type,
            style_guide=proj.style_guide or {},
            locked_glossary=locked_glossary,
            custom_instructions=instruction,
        )
        gate_service = TranslationQualityGate()
        final_translation = ""
        final_gate = gate_service.validate(node.content, "", locked_glossary)

        # Mỗi lần thử đều dịch từ nguồn gốc và kiểm tra trước khi lưu.
        for attempt in range(1, 3):
            issue_text = "\n".join(
                f"- {issue['code']}: {issue['message']}" for issue in final_gate.issues
            )
            retry_prompt = sys_prompt
            if attempt > 1:
                retry_prompt += (
                    "\n\nPrevious translation failed validation. Translate the ORIGINAL SOURCE again.\n"
                    f"Errors:\n{issue_text}\n"
                    "Do not summarize or omit content. Preserve numbers, URLs and references."
                )
            try:
                revised = provider.translate_single(
                    text=node.content,
                    system_prompt=f"{retry_prompt}\n\n{contextual_user_prompt}",
                    glossary_terms=context.glossary,
                    model=model_name,
                    temperature=0.15 if attempt == 1 else 0.0,
                )
            except Exception as exc:
                print(f"[Retranslate] provider.translate_single failed: {exc}")
                revised = ""
            final_translation = VietnamesePostProcessor.normalize_safely(revised or "")
            final_gate = gate_service.validate(node.content, final_translation, locked_glossary)
            if final_gate.passed:
                break

        if not final_gate.passed:
            node.status = "NEEDS_REVIEW"
            db.commit()
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "TRANSLATION_VALIDATION_FAILED",
                    "issues": final_gate.issues,
                },
            )

        trans_repo = TranslationRepository(db)
        trans_repo.save_node_translation(
            node_id=node_id,
            project_id=project_id,
            translated_text=final_translation,
            model_name=model_name,
            instruction=instruction,
            created_by="user_retranslate",
            prompt_version=signature.prompt_version,
        )
        TranslationMemoryService.store(
            db=db,
            source_text=node.content,
            translated_text=final_translation,
            style_hash=signature.style_hash,
            glossary_hash=signature.glossary_hash,
            model_name=model_name,
            prompt_version=signature.prompt_version,
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
