import uuid
from pathlib import Path
from typing import List, Dict, Any

try:
    import docx
except ImportError:
    docx = None


class DOCXExtractor:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

    def extract_document(self, docx_path: Path) -> Dict[str, Any]:
        if docx is None:
            raise ImportError("python-docx is required for DOCX extraction.")

        doc = docx.Document(str(docx_path))
        page_blocks = []
        b_idx = 0

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue

            style_name = p.style.name.lower()
            if "heading 1" in style_name:
                page_blocks.append({
                    "block_id": b_idx,
                    "type": "heading",
                    "text": text,
                    "heading_level": 1,
                    "font_size": 18.0,
                    "is_bold": True,
                    "confidence": 1.0,
                })
            elif "heading 2" in style_name:
                page_blocks.append({
                    "block_id": b_idx,
                    "type": "heading",
                    "text": text,
                    "heading_level": 2,
                    "font_size": 15.0,
                    "is_bold": True,
                    "confidence": 1.0,
                })
            elif "heading 3" in style_name:
                page_blocks.append({
                    "block_id": b_idx,
                    "type": "heading",
                    "text": text,
                    "heading_level": 3,
                    "font_size": 13.0,
                    "is_bold": True,
                    "confidence": 1.0,
                })
            else:
                page_blocks.append({
                    "block_id": b_idx,
                    "type": "text",
                    "text": text,
                    "font_size": 11.0,
                    "is_bold": False,
                    "confidence": 1.0,
                })
            b_idx += 1

        pages = [{
            "page_number": 1,
            "page_type": "TEXT_PAGE",
            "blocks": page_blocks,
            "word_count": sum(len(b.get("text", "").split()) for b in page_blocks if "text" in b),
        }]

        return {
            "page_count": 1,
            "text_pages_count": 1,
            "scanned_pages_count": 0,
            "pages": pages,
            "assets": [],
            "toc": [],
        }
