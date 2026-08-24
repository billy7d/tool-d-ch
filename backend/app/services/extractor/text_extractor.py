import re
from pathlib import Path
from typing import List, Dict, Any


class TextExtractor:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

    def extract_document(self, text_path: Path) -> Dict[str, Any]:
        content = text_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        
        page_blocks = []
        b_idx = 0
        current_para = []

        def flush_para():
            nonlocal current_para, b_idx
            if current_para:
                full_text = " ".join(current_para).strip()
                if full_text:
                    page_blocks.append({
                        "block_id": b_idx,
                        "type": "text",
                        "text": full_text,
                        "font_size": 11.0,
                        "is_bold": False,
                        "confidence": 1.0,
                    })
                    b_idx += 1
                current_para = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                flush_para()
                continue

            # Markdown Heading check
            h_match = re.match(r"^(#{1,6})\s+(.*)$", line_str)
            if h_match:
                flush_para()
                lvl = len(h_match.group(1))
                page_blocks.append({
                    "block_id": b_idx,
                    "type": "heading",
                    "text": h_match.group(2).strip(),
                    "heading_level": min(lvl, 4),
                    "font_size": 18.0 - (lvl * 2),
                    "is_bold": True,
                    "confidence": 1.0,
                })
                b_idx += 1
            else:
                current_para.append(line_str)

        flush_para()

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
