from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from typing import List

from app.db.engine import get_project_db
from app.db.models import NodeModel
from app.db.repository import GlossaryRepository
from app.models.schemas import GlossaryItemCreate, GlossaryItemUpdate, GlossaryItemResponse
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.worker import translation_worker

router = APIRouter(prefix="/api/projects/{project_id}/glossary", tags=["Glossary"])


@router.get("", response_model=List[GlossaryItemResponse])
def list_glossary_terms(project_id: str):
    db = get_project_db(project_id)
    repo = GlossaryRepository(db)
    try:
        items = repo.list_glossary(project_id)
        return [
            GlossaryItemResponse(
                id=i.id,
                source_term=i.source_term,
                target_term=i.target_term,
                category=i.category,
                notes=i.notes,
                locked=i.locked,
                created_at=i.created_at
            ) for i in items
        ]
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
            locked=payload.locked
        )
        return GlossaryItemResponse(
            id=item.id,
            source_term=item.source_term,
            target_term=item.target_term,
            category=item.category,
            notes=item.notes,
            locked=item.locked,
            created_at=item.created_at
        )
    finally:
        db.close()


@router.patch("/{term_id}", response_model=GlossaryItemResponse)
def update_glossary_term(project_id: str, term_id: str, payload: GlossaryItemUpdate):
    db = get_project_db(project_id)
    repo = GlossaryRepository(db)
    try:
        item = repo.update_term(term_id, **payload.model_dump(exclude_unset=True))
        if not item:
            raise HTTPException(status_code=404, detail="Không tìm thấy thuật ngữ.")
        return GlossaryItemResponse(
            id=item.id,
            source_term=item.source_term,
            target_term=item.target_term,
            category=item.category,
            notes=item.notes,
            locked=item.locked,
            created_at=item.created_at
        )
    finally:
        db.close()


@router.delete("/{term_id}")
def delete_glossary_term(project_id: str, term_id: str):
    db = get_project_db(project_id)
    repo = GlossaryRepository(db)
    try:
        success = repo.delete_term(term_id)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy thuật ngữ.")
        return {"message": "Đã xóa thuật ngữ."}
    finally:
        db.close()


@router.post("/extract_auto")
def auto_extract_glossary(project_id: str):
    db = get_project_db(project_id)
    try:
        nodes = db.query(NodeModel).filter(NodeModel.project_id == project_id).limit(20).all()
        sample_text = "\n".join([n.content for n in nodes])
        provider = translation_worker.get_provider("qwen2.5:7b")
        extracted = provider.extract_glossary(sample_text)

        repo = GlossaryRepository(db)
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
                added_count += 1

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
        return {"message": f"Đã nhập thành công {count} thuật ngữ.", "count": count}
    finally:
        db.close()
