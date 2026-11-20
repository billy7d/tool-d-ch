from pathlib import Path
from typing import Optional, Tuple

from app.models.canonical import CanonicalDocument
from app.services.ebook.book_plan import BookStyle, build_book_plan
from app.services.ebook.devices import get_device
from app.services.ebook.epub_writer import write_epub
from app.services.ebook.mobi_writer import write_mobi
from app.services.ebook.pdf_to_ebook import calibre_executable, convert_epub_to_mobi_with_calibre
from app.services.exporter.validator import ExportValidator


class MOBIExporter:
    @classmethod
    def export_mobi(
        cls,
        doc: CanonicalDocument,
        output_mobi_path: Path,
        metadata: Optional[dict] = None,
        device: Optional[str] = None,
        cover_image: Optional[bytes] = None,
        style: Optional[BookStyle] = None,
        engine: str = "auto",
    ) -> Path:
        return cls.export_mobi_with_engine(doc, output_mobi_path, metadata, device, cover_image, style, engine)[0]

    @classmethod
    def export_mobi_with_engine(
        cls,
        doc: CanonicalDocument,
        output_mobi_path: Path,
        metadata: Optional[dict] = None,
        device: Optional[str] = None,
        cover_image: Optional[bytes] = None,
        style: Optional[BookStyle] = None,
        engine: str = "auto",
    ) -> Tuple[Path, str]:
        """MOBI 6 cho Kindle đời cũ. Có Calibre thì dùng Calibre (--mobi-file-type old), không thì bộ ghi tích hợp."""
        metadata = metadata or {}
        output_mobi_path = Path(output_mobi_path)
        output_mobi_path.parent.mkdir(parents=True, exist_ok=True)
        profile = get_device(device)
        plan = build_book_plan(
            doc,
            language=metadata.get("language"),
            style=style,
            cover=cover_image,
            title=metadata.get("title"),
            author=metadata.get("author"),
            translator=metadata.get("translator"),
        )

        used = "builtin"
        if engine == "calibre" or (engine == "auto" and calibre_executable()):
            temp_epub = output_mobi_path.parent / f"{output_mobi_path.stem}_temp.epub"
            try:
                write_epub(plan, temp_epub, profile, grayscale_images=True)
                convert_epub_to_mobi_with_calibre(temp_epub, output_mobi_path, profile, plan.language)
                used = "calibre"
            except Exception:
                if engine == "calibre":
                    raise
                write_mobi(plan, output_mobi_path, profile)
            finally:
                temp_epub.unlink(missing_ok=True)
        else:
            write_mobi(plan, output_mobi_path, profile)

        valid, msg = ExportValidator.validate_export_file(output_mobi_path, "mobi")
        if not valid:
            raise RuntimeError(f"Lỗi xuất MOBI: {msg}")
        return output_mobi_path, used
