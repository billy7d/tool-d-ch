import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime

from app.db.engine import get_project_db
from app.db.models import LayoutProfileModel, ExportModel
from app.db.repository import StructureRepository
from app.models.schemas import ExportRequest, ExportResponse
from app.services.project_manager import ProjectFileManager
from app.services.exporter.pdf_exporter import PDFExporter
from app.services.exporter.epub_exporter import EPUBExporter
from app.services.exporter.mobi_exporter import MOBIExporter

router = APIRouter(prefix="/api/projects/{project_id}/export", tags=["Export"])


@router.post("", response_model=ExportResponse)
def export_document(project_id: str, payload: ExportRequest):
    db = get_project_db(project_id)
    struct_repo = StructureRepository(db)
    
    try:
        doc = struct_repo.get_canonical_document(project_id)
        if payload.title:
            doc.metadata.title = payload.title
        if payload.author:
            doc.metadata.author = payload.author
        if payload.translator:
            doc.metadata.translator = payload.translator

        # Get layout profile
        profile = None
        if payload.layout_profile_id:
            profile = db.query(LayoutProfileModel).filter(LayoutProfileModel.id == payload.layout_profile_id).first()
        if not profile:
            profile = db.query(LayoutProfileModel).filter(LayoutProfileModel.project_id == project_id).first()
        if not profile:
            profile = LayoutProfileModel(
                id=str(uuid.uuid4()),
                project_id=project_id,
                name="Classic Book"
            )

        export_dir = ProjectFileManager.get_project_dir(project_id) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        fmt = payload.format.lower()
        export_id = str(uuid.uuid4())
        safe_title = (doc.metadata.title or "Document").replace(" ", "_")

        if fmt == "pdf":
            out_file = export_dir / f"{safe_title}_{export_id[:8]}.pdf"
            PDFExporter.export_pdf(doc, profile, out_file)
        elif fmt == "epub":
            out_file = export_dir / f"{safe_title}_{export_id[:8]}.epub"
            EPUBExporter.export_epub(doc, out_file)
        elif fmt == "mobi":
            out_file = export_dir / f"{safe_title}_{export_id[:8]}.mobi"
            MOBIExporter.export_mobi(doc, out_file)
        else:
            raise HTTPException(status_code=400, detail=f"Định dạng xuất '{fmt}' không được hỗ trợ.")

        file_size = out_file.stat().st_size

        # Record export
        exp_rec = ExportModel(
            id=export_id,
            project_id=project_id,
            export_format=fmt,
            file_path=str(out_file),
            file_size_bytes=file_size,
            status="COMPLETED"
        )
        db.add(exp_rec)
        db.commit()

        return ExportResponse(
            export_id=export_id,
            export_format=fmt,
            download_url=f"/api/projects/{project_id}/export/download/{export_id}",
            file_size_bytes=file_size,
            created_at=datetime.utcnow()
        )
    finally:
        db.close()


@router.get("/download/{export_id}")
def download_export(project_id: str, export_id: str):
    db = get_project_db(project_id)
    try:
        exp_rec = db.query(ExportModel).filter(ExportModel.id == export_id).first()
        if not exp_rec or not Path(exp_rec.file_path).exists():
            raise HTTPException(status_code=404, detail="File xuất bản không tìm thấy hoặc đã bị xóa.")

        file_path = Path(exp_rec.file_path)
        media_types = {
            "pdf": "application/pdf",
            "epub": "application/epub+zip",
            "mobi": "application/x-mobipocket-ebook",
        }
        media_type = media_types.get(exp_rec.export_format, "application/octet-stream")

        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type=media_type
        )
    finally:
        db.close()
