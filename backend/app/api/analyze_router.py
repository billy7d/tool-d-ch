from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from typing import Dict, Any

from app.db.engine import get_project_db
from app.db.repository import DocumentRepository, ProjectRepository, StructureRepository
from app.services.project_manager import ProjectFileManager
from app.services.extractor.pdf_extractor import PDFExtractor
from app.services.extractor.epub_extractor import EPUBExtractor
from app.services.extractor.docx_extractor import DOCXExtractor
from app.services.extractor.text_extractor import TextExtractor
from app.services.reconstruction.structure_builder import StructureBuilder
from app.services.translation.worker import translation_worker

router = APIRouter(prefix="/api/projects/{project_id}/analyze", tags=["Analysis"])


def _run_analysis_pipeline(project_id: str):
    db = get_project_db(project_id)
    doc_repo = DocumentRepository(db)
    proj_repo = ProjectRepository(db)
    struct_repo = StructureRepository(db)

    try:
        project_dir = ProjectFileManager.get_project_dir(project_id)
        docs = doc_repo.get_source_documents(project_id)
        if not docs:
            return

        doc_record = docs[0]
        file_path = Path(doc_record.file_path)
        ext = doc_record.file_format.lower()

        translation_worker.broadcast_event("ANALYSIS_STARTED", {"project_id": project_id})

        # 1. Extraction based on format
        if ext == "pdf":
            extractor = PDFExtractor(project_dir)
            extracted_data = extractor.extract_document(file_path)
        elif ext == "epub":
            extractor = EPUBExtractor(project_dir)
            extracted_data = extractor.extract_document(file_path)
        elif ext == "docx":
            extractor = DOCXExtractor(project_dir)
            extracted_data = extractor.extract_document(file_path)
        else:
            extractor = TextExtractor(project_dir)
            extracted_data = extractor.extract_document(file_path)

        # Update source document stats
        doc_record.page_count = extracted_data["page_count"]
        doc_record.text_pages_count = extracted_data.get("text_pages_count", 0)
        doc_record.scanned_pages_count = extracted_data.get("scanned_pages_count", 0)
        doc_record.analysis_status = "COMPLETED"
        db.commit()

        # 2. Structure Reconstruction
        canonical_doc = StructureBuilder.build_canonical_document(
            project_id=project_id,
            filename=doc_record.filename,
            pages_data=extracted_data["pages"],
            assets_data=extracted_data["assets"],
            source_toc=extracted_data.get("toc")
        )

        # 3. Save Canonical Model into Database
        struct_repo.save_canonical_document(project_id, canonical_doc)

        proj_repo.update_project(project_id, current_stage="STRUCTURE_READY")
        translation_worker.broadcast_event("ANALYSIS_COMPLETED", {"project_id": project_id})

    finally:
        db.close()


@router.post("")
def start_analysis(project_id: str, background_tasks: BackgroundTasks):
    db = get_project_db(project_id)
    proj_repo = ProjectRepository(db)
    proj = proj_repo.get_project(project_id)
    if not proj:
        db.close()
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
    db.close()

    background_tasks.add_task(_run_analysis_pipeline, project_id)
    return {"message": "Đang tiến hành phân tích tài liệu và nhận diện cấu trúc."}
