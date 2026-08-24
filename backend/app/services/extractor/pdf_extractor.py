import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import io

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from app.services.extractor.scan_detector import ScanDetector, PageType
from app.services.extractor.ocr_engine import OCREngine


class PDFExtractor:
    def __init__(self, project_dir: Path, ocr_lang: str = "eng"):
        self.project_dir = project_dir
        self.thumbnails_dir = project_dir / "cache" / "page_thumbnails"
        self.assets_dir = project_dir / "assets" / "original"
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_engine = OCREngine(lang=ocr_lang)

    def extract_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extracts structured content from a PDF document.
        Returns:
        {
            "page_count": int,
            "text_pages_count": int,
            "scanned_pages_count": int,
            "pages": List[Dict], # Raw structured pages
            "assets": List[Dict], # Extracted images
            "toc": List[Dict], # Source PDF Table of Contents
        }
        """
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for PDF extraction.")

        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        
        # Source TOC if available
        toc_entries = []
        try:
            raw_toc = doc.get_toc()  # [[lvl, title, page], ...]
            for item in raw_toc:
                if len(item) >= 3:
                    toc_entries.append({
                        "level": item[0],
                        "title": item[1],
                        "page": item[2]
                    })
        except Exception:
            pass

        pages_data = []
        assets_data = []
        text_pages_count = 0
        scanned_pages_count = 0

        for page_idx in range(page_count):
            page = doc[page_idx]
            page_num = page_idx + 1
            rect = page.rect
            page_area = rect.width * rect.height

            # Generate and save thumbnail (WebP/JPEG for fast preview)
            thumb_path = self.thumbnails_dir / f"page_{page_num:06d}.jpg"
            if not thumb_path.exists():
                pix = page.get_pixmap(dpi=150)
                pix.save(str(thumb_path))

            # Extract text blocks and spans with font metadata
            text_page = page.get_text("dict")
            blocks = text_page.get("blocks", [])
            
            raw_text = page.get_text()
            char_count = len(raw_text.strip())
            word_count = len(raw_text.strip().split())

            # Detect images on page
            image_list = page.get_images(full=True)
            image_count = len(image_list)
            
            # Estimate image area
            image_area = 0.0
            for img in image_list:
                try:
                    # Calculate bbox if possible
                    bbox = page.get_image_bbox(img)
                    image_area += (bbox.width * bbox.height)
                except Exception:
                    pass

            image_area_ratio = (image_area / page_area) if page_area > 0 else 0.0

            # Scan classification
            page_type = ScanDetector.classify_page(
                text_char_count=char_count,
                word_count=word_count,
                image_count=image_count,
                image_area_ratio=image_area_ratio,
                page_area=page_area
            )

            if page_type == PageType.SCANNED_PAGE:
                scanned_pages_count += 1
            else:
                text_pages_count += 1

            page_blocks = []

            # If SCANNED_PAGE or MIXED_PAGE with poor text, perform selective OCR
            if page_type in (PageType.SCANNED_PAGE, PageType.MIXED_PAGE) and char_count < 100 and self.ocr_engine.is_available:
                # Render high-res image for OCR
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_blocks = self.ocr_engine.ocr_image(img)
                
                for ob_idx, ob in enumerate(ocr_blocks):
                    page_blocks.append({
                        "block_id": ob_idx,
                        "type": "text",
                        "text": ob["text"],
                        "bbox": ob["bbox"],
                        "font_size": 11.0,
                        "font_name": "OCR_Detected",
                        "is_bold": False,
                        "is_italic": False,
                        "confidence": ob["confidence"],
                        "is_ocr": True
                    })
            else:
                # Process native PDF blocks
                for b_idx, block in enumerate(blocks):
                    if block.get("type") == 0:  # Text block
                        block_text_parts = []
                        font_sizes = []
                        font_names = []
                        is_bold = False
                        is_italic = False

                        for line in block.get("lines", []):
                            line_text_parts = []
                            for span in line.get("spans", []):
                                stext = span.get("text", "")
                                if stext:
                                    line_text_parts.append(stext)
                                    font_sizes.append(span.get("size", 11.0))
                                    font_names.append(span.get("font", ""))
                                    flags = span.get("flags", 0)
                                    if flags & 2 or "Bold" in span.get("font", ""):
                                        is_bold = True
                                    if flags & 1 or "Italic" in span.get("font", ""):
                                        is_italic = True
                            
                            line_text = "".join(line_text_parts)
                            if line_text:
                                block_text_parts.append(line_text)

                        full_text = " ".join(block_text_parts).strip()
                        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 11.0
                        dom_font = font_names[0] if font_names else "Default"

                        if full_text:
                            page_blocks.append({
                                "block_id": b_idx,
                                "type": "text",
                                "text": full_text,
                                "bbox": list(block.get("bbox", [0, 0, 0, 0])),
                                "font_size": round(avg_font_size, 1),
                                "font_name": dom_font,
                                "is_bold": is_bold,
                                "is_italic": is_italic,
                                "confidence": 1.0,
                                "is_ocr": False
                            })

                    elif block.get("type") == 1:  # Image block
                        asset_id = str(uuid.uuid4())
                        img_filename = f"img_{asset_id[:8]}.png"
                        img_path = self.assets_dir / img_filename
                        
                        try:
                            # Save image bytes
                            image_bytes = block.get("image")
                            if image_bytes:
                                with open(img_path, "wb") as f:
                                    f.write(image_bytes)
                                
                                asset_record = {
                                    "id": asset_id,
                                    "original_path": str(img_path),
                                    "source_page": page_num,
                                    "width": block.get("width"),
                                    "height": block.get("height"),
                                    "mime_type": "image/png"
                                }
                                assets_data.append(asset_record)
                                
                                page_blocks.append({
                                    "block_id": b_idx,
                                    "type": "image",
                                    "asset_id": asset_id,
                                    "bbox": list(block.get("bbox", [0, 0, 0, 0])),
                                    "confidence": 1.0,
                                    "is_ocr": False
                                })
                        except Exception as e:
                            print(f"Error extracting image on page {page_num}: {e}")

            pages_data.append({
                "page_number": page_num,
                "page_type": page_type.value,
                "width": rect.width,
                "height": rect.height,
                "thumbnail_path": str(thumb_path),
                "blocks": page_blocks,
                "word_count": word_count,
            })

        doc.close()

        return {
            "page_count": page_count,
            "text_pages_count": text_pages_count,
            "scanned_pages_count": scanned_pages_count,
            "pages": pages_data,
            "assets": assets_data,
            "toc": toc_entries,
        }
