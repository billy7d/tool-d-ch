import io
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.db.engine import get_global_db, get_project_db
from app.db.repository import ProjectRepository
from app.models.schemas import ProjectCreate, ProjectUpdate, ProjectResponse
from app.services.project_manager import ProjectFileManager

router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.post("", response_model=ProjectResponse)
def create_project(payload: ProjectCreate, db: Session = Depends(get_global_db)):
    repo = ProjectRepository(db)
    proj = repo.create_project(
        title=payload.title,
        description=payload.description,
        source_language=payload.source_language,
        target_language=payload.target_language,
        document_type=payload.document_type.value,
        translation_mode=payload.translation_mode.value,
        selected_model=payload.selected_model,
    )
    # Initialize project directory and database with the exact same project ID
    ProjectFileManager.init_project_structure(proj.id)
    proj_db = get_project_db(proj.id)
    proj_db_repo = ProjectRepository(proj_db)
    proj_db_repo.create_project(
        project_id=proj.id,
        title=payload.title,
        description=payload.description,
        source_language=payload.source_language,
        target_language=payload.target_language,
        document_type=payload.document_type.value,
        translation_mode=payload.translation_mode.value,
        selected_model=payload.selected_model,
    )
    proj_db.close()

    stats = repo.get_project_stats(proj.id)
    return ProjectResponse(
        id=proj.id,
        title=proj.title,
        description=proj.description,
        source_language=proj.source_language,
        target_language=proj.target_language,
        document_type=proj.document_type,
        translation_mode=proj.translation_mode,
        custom_instructions=proj.custom_instructions,
        current_stage=proj.current_stage,
        structure_version=proj.structure_version,
        structure_confirmed=proj.structure_confirmed,
        selected_model=proj.selected_model,
        qa_level=proj.qa_level,
        style_guide=proj.style_guide,
        created_at=proj.created_at,
        updated_at=proj.updated_at,
        **stats
    )


@router.get("", response_model=List[ProjectResponse])
def list_projects(db: Session = Depends(get_global_db)):
    repo = ProjectRepository(db)
    projects = repo.list_projects()
    res = []
    for p in projects:
        stats = repo.get_project_stats(p.id)
        res.append(ProjectResponse(
            id=p.id,
            title=p.title,
            description=p.description,
            source_language=p.source_language,
            target_language=p.target_language,
            document_type=p.document_type,
            translation_mode=p.translation_mode,
            custom_instructions=p.custom_instructions,
            current_stage=p.current_stage,
            structure_version=p.structure_version,
            structure_confirmed=p.structure_confirmed,
            selected_model=p.selected_model,
            qa_level=p.qa_level,
            style_guide=p.style_guide,
            created_at=p.created_at,
            updated_at=p.updated_at,
            **stats
        ))
    return res


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_global_db)):
    repo = ProjectRepository(db)
    p = repo.get_project(project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")
    stats = repo.get_project_stats(p.id)
    return ProjectResponse(
        id=p.id,
        title=p.title,
        description=p.description,
        source_language=p.source_language,
        target_language=p.target_language,
        document_type=p.document_type,
        translation_mode=p.translation_mode,
        custom_instructions=p.custom_instructions,
        current_stage=p.current_stage,
        structure_version=p.structure_version,
        structure_confirmed=p.structure_confirmed,
        selected_model=p.selected_model,
        qa_level=p.qa_level,
        style_guide=p.style_guide,
        created_at=p.created_at,
        updated_at=p.updated_at,
        **stats
    )


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: str, payload: ProjectUpdate, db: Session = Depends(get_global_db)):
    repo = ProjectRepository(db)
    update_data = payload.model_dump(exclude_unset=True)
    if "translation_mode" in update_data and update_data["translation_mode"]:
        update_data["translation_mode"] = update_data["translation_mode"].value
    if "document_type" in update_data and update_data["document_type"]:
        update_data["document_type"] = update_data["document_type"].value
    if "qa_level" in update_data and update_data["qa_level"]:
        update_data["qa_level"] = update_data["qa_level"].value

    p = repo.update_project(project_id, **update_data)
    if not p:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án.")

    # Sync with project db
    try:
        pdb = get_project_db(project_id)
        prepo = ProjectRepository(pdb)
        prepo.update_project(project_id, **update_data)
        pdb.close()
    except Exception:
        pass

    stats = repo.get_project_stats(p.id)
    return ProjectResponse(
        id=p.id,
        title=p.title,
        description=p.description,
        source_language=p.source_language,
        target_language=p.target_language,
        document_type=p.document_type,
        translation_mode=p.translation_mode,
        custom_instructions=p.custom_instructions,
        current_stage=p.current_stage,
        structure_version=p.structure_version,
        structure_confirmed=p.structure_confirmed,
        selected_model=p.selected_model,
        qa_level=p.qa_level,
        style_guide=p.style_guide,
        created_at=p.created_at,
        updated_at=p.updated_at,
        **stats
    )


@router.delete("/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_global_db)):
    repo = ProjectRepository(db)
    success = repo.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy dự án để xóa.")
    ProjectFileManager.delete_project_files(project_id)
    return {"message": "Đã xóa dự án thành công."}


@router.get("/{project_id}/backup")
def backup_project(project_id: str):
    try:
        zip_path = ProjectFileManager.create_project_backup(project_id)
        return FileResponse(
            path=str(zip_path),
            filename=f"project_{project_id}.project.zip",
            media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi sao lưu dự án: {e}")


@router.post("/restore")
async def restore_project(file: UploadFile = File(...), db: Session = Depends(get_global_db)):
    content = await file.read()
    import uuid
    new_project_id = str(uuid.uuid4())
    try:
        ProjectFileManager.restore_project_from_zip(content, new_project_id)
        # Register in global db
        repo = ProjectRepository(db)
        proj = repo.create_project(
            project_id=new_project_id,
            title=file.filename.replace(".project.zip", ""),
            description="Phục hồi từ bản sao lưu"
        )
        return {"project_id": proj.id, "message": "Phục hồi dự án thành công."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phục hồi dự án: {e}")
