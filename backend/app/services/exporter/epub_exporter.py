from pathlib import Path
from typing import Optional

from app.models.canonical import CanonicalDocument
from app.services.ebook.book_plan import BookStyle, build_book_plan
from app.services.ebook.devices import get_device
from app.services.ebook.epub_writer import write_epub
from app.services.exporter.validator import ExportValidator


class EPUBExporter:
    @classmethod
    def export_epub(
        cls,
        doc: CanonicalDocument,
        output_epub_path: Path,
        metadata: Optional[dict] = None,
        device: Optional[str] = None,
        cover_image: Optional[bytes] = None,
        style: Optional[BookStyle] = None,
        grayscale_images: bool = False,
    ) -> Path:
        """EPUB 3 (kèm toc.ncx của EPUB 2) tối ưu cho KOReader: ảnh nhúng thật, mục lục nhiều cấp, CSS theo em."""
        metadata = metadata or {}
        plan = build_book_plan(
            doc,
            language=metadata.get("language"),
            style=style,
            cover=cover_image,
            title=metadata.get("title"),
            author=metadata.get("author"),
            translator=metadata.get("translator"),
        )
        write_epub(plan, Path(output_epub_path), get_device(device), grayscale_images=grayscale_images)

        valid, msg = ExportValidator.validate_export_file(Path(output_epub_path), "epub")
        if not valid:
            raise RuntimeError(f"Lỗi xuất EPUB: {msg}")
        return Path(output_epub_path)
