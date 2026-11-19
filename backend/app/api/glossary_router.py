from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from typing import List

from app.db.engine import get_project_db
from app.db.models import GlossaryModel, NodeModel, ProjectModel
from app.db.repository import GlossaryRepository
from app.models.schemas import GlossaryItemCreate, GlossaryItemUpdate, GlossaryItemResponse
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.model_defaults import default_model
from app.services.translation.term_consistency import AUTO_CATEGORY, choose_terms
from app.services.translation.worker import translation_worker
from app.services.translation.semantic_assurance import SemanticAssuranceService

router = APIRouter(prefix="/api/projects/{project_id}/glossary", tags=["Glossary"])


def _glossary_response(item: GlossaryModel) -> GlossaryItemResponse:
    """Dựng response mới nhưng vẫn trả đủ các trường glossary legacy."""
    context = GlossaryService.to_context_entry(item)
    return GlossaryItemResponse(
        id=item.id,
        source_term=item.source_term,
        target_term=item.target_term,
        category=item.category,
        notes=item.notes,
        locked=context["locked"],
        preferred_target=context["preferred_target"],
        allowed_variants=context["allowed_variants"],
        sense_hint=context["sense_hint"],
        domain=context["domain"],
        part_of_speech=context["part_of_speech"] or None,
        preserve_original=context["preserve_original"],
        lock_level=context["lock_level"],
        created_at=item.created_at,
    )


@router.get("", response_model=List[GlossaryItemResponse])
def list_glossary_terms(project_id: str):
    db = get_project_db(project_id)
    repo = GlossaryRepository(db)
    try:
        items = repo.list_glossary(project_id)
        return [_glossary_response(item) for item in items]
    finally:
        db.close()


@router.post("", response_model=GlossaryItemResponse)
def add_glossary_term(project_id: str, payload: GlossaryItemCreate):
    db = get_project_db(project_id)
    repo = GlossaryRepository(db)
    try:
        item = repo.add_term(
            project_id=project_id,
            source_term=payload.source_term,
            target_term=payload.target_term,
            category=payload.category,
            notes=payload.notes,
            locked=payload.locked,
            preferred_target=payload.preferred_target,
            allowed_variants=payload.allowed_variants,
            sense_hint=payload.sense_hint,
            domain=payload.domain,
            part_of_speech=payload.part_of_speech,
            preserve_original=payload.preserve_original,
            lock_level=payload.lock_level,
        )
        SemanticAssuranceService.invalidate_for_source_term(db, project_id, item.source_term)
        db.commit()
        return _glossary_response(item)
    finally:
        db.close()


@router.patch("/{term_id}", response_model=GlossaryItemResponse)
def update_glossary_term(project_id: str, term_id: str, payload: GlossaryItemUpdate):
    db = get_project_db(project_id)
    repo = GlossaryRepository(db)
    try:
        previous = db.query(GlossaryModel).filter(GlossaryModel.id == term_id).first()
        previous_source = previous.source_term if previous else ""
        item = repo.update_term(term_id, **payload.model_dump(exclude_unset=True))
        if not item:
            raise HTTPException(status_code=404, detail="Không tìm thấy thuật ngữ.")
        if previous_source:
            SemanticAssuranceService.invalidate_for_source_term(db, project_id, previous_source)
        SemanticAssuranceService.invalidate_for_source_term(db, project_id, item.source_term)
        db.commit()
        return _glossary_response(item)
    finally:
        db.close()


@router.delete("/{term_id}")
def delete_glossary_term(project_id: str, term_id: str):
    db = get_project_db(project_id)
    repo = GlossaryRepository(db)
    try:
        previous = db.query(GlossaryModel).filter(GlossaryModel.id == term_id).first()
        previous_source = previous.source_term if previous else ""
        success = repo.delete_term(term_id)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy thuật ngữ.")
        if previous_source:
            SemanticAssuranceService.invalidate_for_source_term(db, project_id, previous_source)
            db.commit()
        return {"message": "Đã xóa thuật ngữ."}
    finally:
        db.close()


@router.post("/extract_auto")
def auto_extract_glossary(project_id: str):
    db = get_project_db(project_id)
    try:
        project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        model_name = (project.selected_model if project else None) or default_model()
        provider = translation_worker.get_provider(model_name)
        repo = GlossaryRepository(db)
        if getattr(provider, "json_task", None) is not None:
            # Quét cả cuốn thay vì 20 đoạn đầu, và chỉ thêm dạng gợi ý MỀM: model 7B đề xuất sai khoảng một phần ba,
            # nên người dùng duyệt lại trong bảng thuật ngữ; thuật ngữ đã có thì giữ nguyên.
            rows = db.query(NodeModel.node_type, NodeModel.content).filter(
                NodeModel.project_id == project_id, NodeModel.node_type.in_(["paragraph", "heading"]),
            ).order_by(NodeModel.order_index).all()
            proposed = choose_terms(
                provider, model_name,
                [content for node_type, content in rows if node_type == "paragraph" and content],
                [content for node_type, content in rows if node_type == "heading" and content],
            )
            existing = {str(term.source_term).casefold() for term in repo.list_glossary(project_id)}
            added_count = 0
            for term in proposed:
                if term["source_term"].casefold() in existing:
                    continue
                repo.add_term(
                    project_id=project_id, source_term=term["source_term"], target_term=term["target_term"],
                    category=AUTO_CATEGORY, notes="Gợi ý tự động, cần duyệt lại.", locked=False, lock_level="SOFT",
                )
                SemanticAssuranceService.invalidate_for_source_term(db, project_id, term["source_term"])
                added_count += 1
            db.commit()
            return {"message": f"Đã đề xuất {added_count} thuật ngữ dạng gợi ý. Hãy duyệt lại trước khi dịch.", "count": added_count}

        nodes = db.query(NodeModel).filter(NodeModel.project_id == project_id).limit(20).all()
        sample_text = "\n".join([n.content for n in nodes])
        extracted = provider.extract_glossary(sample_text)

        added_count = 0
        for term in extracted:
            src = term.get("source_term", "").strip()
            tgt = term.get("target_term", "").strip()
            if src and tgt:
                repo.add_term(
                    project_id=project_id,
                    source_term=src,
                    target_term=tgt,
                    category=term.get("category", "GENERAL"),
                    notes=term.get("notes"),
                    locked=True
                )
                SemanticAssuranceService.invalidate_for_source_term(db, project_id, src)
                added_count += 1

        db.commit()

        return {"message": f"Đã tự động trích xuất và thêm {added_count} thuật ngữ vào bảng tra cứu.", "count": added_count}
    finally:
        db.close()


@router.get("/export_csv")
def export_glossary_csv(project_id: str):
    db = get_project_db(project_id)
    try:
        csv_str = GlossaryService.export_csv(db, project_id)
        return Response(
            content=csv_str,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=glossary_{project_id}.csv"}
        )
    finally:
        db.close()


@router.post("/import_csv")
async def import_glossary_csv(project_id: str, file: UploadFile = File(...)):
    content = await file.read()
    csv_str = content.decode("utf-8", errors="ignore")
    db = get_project_db(project_id)
    try:
        count = GlossaryService.import_csv(db, project_id, csv_str)
        # Import có thể thay đổi nhiều term; đánh stale toàn bộ review của project.
        node_ids = [node.id for node in db.query(NodeModel).filter(NodeModel.project_id == project_id).all()]
        SemanticAssuranceService.invalidate_for_nodes(db, project_id, node_ids)
        db.commit()
        return {"message": f"Đã nhập thành công {count} thuật ngữ.", "count": count}
    finally:
        db.close()
