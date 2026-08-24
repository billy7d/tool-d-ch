import os
import uuid
from pathlib import Path
from typing import Optional
from app.models.canonical import CanonicalDocument, Chapter, NodeType
from app.services.renderer.semantic_html import SemanticHTMLRenderer
from app.services.exporter.validator import ExportValidator

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None
    epub = None


class EPUBExporter:
    @classmethod
    def export_epub(
        cls,
        doc: CanonicalDocument,
        output_epub_path: Path,
        metadata: Optional[dict] = None
    ) -> Path:
        if epub is None:
            raise ImportError("Thư viện ebooklib chưa được cài đặt để xuất file EPUB.")

        output_epub_path.parent.mkdir(parents=True, exist_ok=True)
        book = epub.EpubBook()

        # Metadata
        title = doc.metadata.title or "Sách dịch"
        book.set_identifier(doc.id or str(uuid.uuid4()))
        book.set_title(title)
        book.set_language("vi")
        book.add_author(doc.metadata.author or "Tác giả")

        # Custom CSS
        style = """
        body {
            font-family: 'Noto Serif', Georgia, serif;
            line-height: 1.5;
            color: #222;
        }
        h1.chapter-title {
            text-align: center;
            margin-top: 2em;
            margin-bottom: 1em;
        }
        p {
            margin-bottom: 0.5em;
            text-indent: 1.2em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
        }
        th, td {
            border: 1px solid #ccc;
            padding: 6px;
        }
        """
        default_css = epub.EpubItem(
            uid="style_nav",
            file_name="style/nav.css",
            media_type="text/css",
            content=style.encode("utf-8")
        )
        book.add_item(default_css)

        epub_chapters = []
        toc_items = []

        for ch_idx, ch in enumerate(doc.chapters):
            ch_title = ch.translated_title or ch.title
            num_str = f"Chương {ch.number}: " if ch.number else ""
            full_title = num_str + ch_title

            html_body = SemanticHTMLRenderer.render_chapter(ch)
            chapter_item = epub.EpubHtml(
                title=full_title,
                file_name=f"chap_{ch_idx+1:04d}.xhtml",
                lang="vi"
            )
            chapter_item.set_content(f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="vi">
<head>
    <title>{full_title}</title>
    <link rel="stylesheet" href="style/nav.css" type="text/css" />
</head>
<body>
    {html_body}
</body>
</html>""".encode("utf-8"))
            chapter_item.add_item(default_css)

            book.add_item(chapter_item)
            epub_chapters.append(chapter_item)
            toc_items.append(chapter_item)

        # Table of Contents
        book.toc = tuple(toc_items)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Spine
        book.spine = ["nav"] + epub_chapters

        # Write EPUB
        epub.write_epub(str(output_epub_path), book, {})

        # Validate
        valid, msg = ExportValidator.validate_export_file(output_epub_path, "epub")
        if not valid:
            raise RuntimeError(f"Lỗi xuất EPUB: {msg}")

        return output_epub_path
