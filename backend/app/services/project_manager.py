import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from app.config import settings


class ProjectFileManager:
    @staticmethod
    def get_project_dir(project_id: str) -> Path:
        return settings.PROJECTS_DIR / project_id

    @staticmethod
    def init_project_structure(project_id: str) -> Path:
        """Creates the isolated folder structure required by PRD Section 16."""
        pdir = ProjectFileManager.get_project_dir(project_id)
        subdirs = [
            "source",
            "assets/original",
            "assets/derived",
            "cache/page_thumbnails",
            "cache/ocr",
            "cache/preview",
            "render",
            "exports",
            "logs",
        ]
        for s in subdirs:
            (pdir / s).mkdir(parents=True, exist_ok=True)
        return pdir

    @staticmethod
    def save_source_file(project_id: str, filename: str, content_bytes: bytes) -> Path:
        pdir = ProjectFileManager.init_project_structure(project_id)
        target_path = pdir / "source" / filename
        with open(target_path, "wb") as f:
            f.write(content_bytes)
        return target_path

    @staticmethod
    def create_project_backup(project_id: str, include_source: bool = True) -> Path:
        """Packages the project directory into a .project.zip file."""
        pdir = ProjectFileManager.get_project_dir(project_id)
        if not pdir.exists():
            raise FileNotFoundError(f"Project directory {pdir} does not exist.")

        export_dir = pdir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        zip_path = export_dir / f"{project_id}_backup.project.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(pdir):
                # Avoid recursively packaging the exports/backup zip itself
                if "exports" in Path(root).parts and zip_path.name in files:
                    files.remove(zip_path.name)
                if not include_source and "source" in Path(root).parts:
                    continue
                for file in files:
                    file_full = Path(root) / file
                    rel_path = file_full.relative_to(pdir)
                    zipf.write(file_full, arcname=rel_path)

        return zip_path

    @staticmethod
    def restore_project_from_zip(zip_bytes: bytes, target_project_id: str) -> Path:
        pdir = ProjectFileManager.get_project_dir(target_project_id)
        pdir.mkdir(parents=True, exist_ok=True)
        
        temp_zip = pdir / "restore_temp.zip"
        with open(temp_zip, "wb") as f:
            f.write(zip_bytes)
            
        with zipfile.ZipFile(temp_zip, "r") as zipf:
            zipf.extractall(pdir)
            
        if temp_zip.exists():
            temp_zip.unlink()
            
        return pdir

    @staticmethod
    def delete_project_files(project_id: str):
        pdir = ProjectFileManager.get_project_dir(project_id)
        if pdir.exists():
            shutil.rmtree(pdir, ignore_errors=True)
