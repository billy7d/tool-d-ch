import os
import uuid
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None
    epub = None


class EPUBExtractor:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.assets_dir = project_dir / "assets" / "original"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def extract_document(self, epub_path: Path) -> Dict[str, Any]:
        if epub is None:
            raise ImportError("ebooklib is required for EPUB extraction.")

        book = epub.read_epub(str(epub_path))
        pages_data = []
        assets_data = []
        toc_entries = []

        page_num = 1
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                
                # Extract text blocks
                page_blocks = []
                b_idx = 0
                for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "ul", "ol", "table", "img"]):
                    tag = elem.name.lower()
                    
                    if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        lvl = int(tag[1])
                        text = elem.get_text().strip()
                        if text:
                            page_blocks.append({
                                "block_id": b_idx,
                                "type": "heading",
                                "text": text,
                                "heading_level": lvl,
                                "font_size": 18.0 - (lvl * 2),
                                "is_bold": True,
                                "confidence": 1.0,
                            })
                            b_idx += 1
                    elif tag == "p":
                        text = elem.get_text().strip()
                        if text:
                            page_blocks.append({
                                "block_id": b_idx,
                                "type": "text",
                                "text": text,
                                "font_size": 11.0,
                                "is_bold": False,
                                "confidence": 1.0,
                            })
                            b_idx += 1
                    elif tag == "img":
                        src = elem.get("src", "")
                        # Try to find corresponding image item
                        asset_id = str(uuid.uuid4())
                        page_blocks.append({
                            "block_id": b_idx,
                            "type": "image",
                            "asset_id": asset_id,
                            "confidence": 1.0,
                        })
                        b_idx += 1

                if page_blocks:
                    pages_data.append({
                        "page_number": page_num,
                        "page_type": "TEXT_PAGE",
                        "blocks": page_blocks,
                        "word_count": sum(len(b.get("text", "").split()) for b in page_blocks if "text" in b),
                    })
                    page_num += 1

            elif item.get_type() == ebooklib.ITEM_IMAGE:
                asset_id = str(uuid.uuid4())
                img_path = self.assets_dir / f"img_{asset_id[:8]}.png"
                with open(img_path, "wb") as f:
                    f.write(item.get_content())
                assets_data.append({
                    "id": asset_id,
                    "original_path": str(img_path),
                    "source_page": 1,
                    "mime_type": "image/png"
                })

        return {
            "page_count": max(len(pages_data), 1),
            "text_pages_count": len(pages_data),
            "scanned_pages_count": 0,
            "pages": pages_data,
            "assets": assets_data,
            "toc": toc_entries,
        }
