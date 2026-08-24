import shutil
import subprocess
from pathlib import Path
from typing import Optional
from app.models.canonical import CanonicalDocument
from app.services.exporter.epub_exporter import EPUBExporter
from app.services.exporter.validator import ExportValidator


class MOBIExporter:
    @classmethod
    def export_mobi(
        cls,
        doc: CanonicalDocument,
        output_mobi_path: Path,
        metadata: Optional[dict] = None
    ) -> Path:
        """
        Converts Canonical Model -> EPUB -> MOBI using Calibre's ebook-convert.
        If Calibre is not installed, raises friendly descriptive error.
        """
        converter_exe = shutil.which("ebook-convert")
        if not converter_exe:
            raise RuntimeError(
                "MOBI export hiện chưa khả dụng do máy chưa cài Calibre (lệnh 'ebook-convert'). "
                "Hãy cài đặt Calibre hoặc xuất định dạng EPUB để đọc trên Kindle/máy đọc sách."
            )

        output_mobi_path.parent.mkdir(parents=True, exist_ok=True)
        temp_epub_path = output_mobi_path.parent / f"{output_mobi_path.stem}_temp.epub"

        # Generate temporary EPUB
        EPUBExporter.export_epub(doc, temp_epub_path, metadata)

        try:
            # Convert EPUB to MOBI
            cmd = [converter_exe, str(temp_epub_path), str(output_mobi_path)]
            subprocess.run(cmd, check=True, timeout=180)
        finally:
            if temp_epub_path.exists():
                temp_epub_path.unlink()

        valid, msg = ExportValidator.validate_export_file(output_mobi_path, "mobi")
        if not valid:
            raise RuntimeError(f"Lỗi xuất MOBI: {msg}")

        return output_mobi_path
