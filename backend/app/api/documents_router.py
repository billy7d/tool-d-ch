import traceback
from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
from typing import List
from app.db.engine import get_project_db, GlobalSessionLocal
from app.db.models import ProjectModel
from app.db.repository import DocumentRepository, ProjectRepository
from app.models.schemas import DocumentUploadResponse
from app.services.project_manager import ProjectFileManager

router = APIRouter(prefix="/api/projects/{project_id}/documents", tags=["Documents"])


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(project_id: str, file: UploadFile = File(...)):
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ["pdf", "epub", "docx", "txt", "md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file '.{ext}' không được hỗ trợ. Vui lòng chọn PDF, EPUB, DOCX, TXT hoặc Markdown."
        )

    try:
        content_bytes = await file.read()
        file_size = len(content_bytes)

        # Save to project source directory immutably (PRD Section 20)
        saved_path = ProjectFileManager.save_source_file(project_id, filename, content_bytes)

        db = get_project_db(project_id)
        try:
            doc_repo = DocumentRepository(db)
            proj_repo = ProjectRepository(db)

            # Estimate preliminary stats
            page_count = 1
            word_count = 0
            if ext == "pdf":
                try:
                    import pymupdf
                    doc = pymupdf.open(str(saved_path))
                    page_count = len(doc)
                    doc.close()
                except Exception as ex:
                    print(f"[Upload] pymupdf warning: {ex}")

            # Add source document record
            doc_record = doc_repo.add_source_document(
                project_id=project_id,
                filename=filename,
                file_path=str(saved_path),
                file_format=ext,
                file_size=file_size,
                page_count=page_count,
                word_count=word_count,
            )

            # Extract fields before session closes to prevent DetachedInstanceError
            res_document_id = doc_record.id
            res_filename = doc_record.filename
            res_format = doc_record.file_format
            res_size = doc_record.file_size_bytes
            res_pages = doc_record.page_count
            res_words = doc_record.word_count_est

            title_from_file = filename.replace(f".{ext}", "").replace("_", " ").title()
            proj_repo.update_project(project_id, title=title_from_file)
        finally:
            db.close()

        # Also update global database title
        try:
            gdb = GlobalSessionLocal()
            g_proj_repo = ProjectRepository(gdb)
            g_proj_repo.update_project(project_id, title=title_from_file)
            gdb.close()
        except Exception:
            pass

        return DocumentUploadResponse(
            document_id=res_document_id,
            filename=res_filename,
            file_format=res_format,
            file_size_bytes=res_size,
            page_count=res_pages,
            word_count_est=res_words,
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lưu trữ tài liệu: {str(e)}"
        )


@router.get("")
def list_documents(project_id: str):
    db = get_project_db(project_id)
    try:
        doc_repo = DocumentRepository(db)
        docs = doc_repo.get_source_documents(project_id)
        # Convert to dict to prevent DetachedInstanceError
        return [
            {
                "id": d.id,
                "project_id": d.project_id,
                "filename": d.filename,
                "file_path": d.file_path,
                "file_format": d.file_format,
                "file_size_bytes": d.file_size_bytes,
                "page_count": d.page_count,
                "word_count_est": d.word_count_est,
                "order_index": d.order_index,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]
    finally:
        db.close()
