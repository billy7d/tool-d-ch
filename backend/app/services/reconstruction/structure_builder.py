import re
import uuid
from typing import List, Dict, Any, Tuple, Optional
from app.models.canonical import (
    CanonicalDocument,
    Chapter,
    DocumentNode,
    NodeType,
    NodeStatus,
    ApprovalStatus,
    NodeMetadata,
    SourceMapping,
    Asset,
    DocumentMetadata,
)
from app.services.reconstruction.paragraph_repair import ParagraphRepair
from app.services.reconstruction.header_footer_filter import HeaderFooterFilter


class StructureBuilder:
    CHAPTER_PATTERNS = [
        re.compile(r"^(?:CHAPTER|Chapter|Ch\.)\s+([0-9IVXLCDM]+|[A-Za-z]+)(?:\s*[:\.\-—]\s*(.*))?$", re.IGNORECASE),
        re.compile(r"^(?:PART|Part)\s+([0-9IVXLCDM]+)(?:\s*[:\.\-—]\s*(.*))?$", re.IGNORECASE),
        re.compile(r"^(?:APPENDIX|Appendix)\s+([0-9A-Z]+)(?:\s*[:\.\-—]\s*(.*))?$", re.IGNORECASE),
        re.compile(r"^([0-9]{1,2})\.\s+([A-Z][A-Za-z0-9\s,\-—]{3,60})$"),
    ]

    FOOTNOTE_PATTERN = re.compile(r"^(?:\[([0-9]{1,3})\]|\*|\†|([0-9]{1,3})\.)\s+(.*)$")

    @classmethod
    def detect_chapter_heading(cls, text: str, font_size: float, is_bold: bool, base_font_size: float = 11.0) -> Optional[Tuple[str, str, int]]:
        """
        Returns (chapter_number, chapter_title, level) if text is detected as a chapter start.
        """
        text_clean = text.strip()
        if not text_clean or len(text_clean) > 150:
            return None

        # Check regex patterns
        for pattern in cls.CHAPTER_PATTERNS:
            m = pattern.match(text_clean)
            if m:
                num = m.group(1) if len(m.groups()) >= 1 else ""
                title = m.group(2) if len(m.groups()) >= 2 and m.group(2) else text_clean
                return (num, title, 1)

        # Check font prominence if all caps or significantly larger than base font
        if font_size >= base_font_size * 1.5 and is_bold and len(text_clean.split()) <= 10:
            if text_clean.isupper() or len(text_clean) < 50:
                return ("", text_clean, 1)

        return None

    @classmethod
    def detect_heading_level(cls, font_size: float, is_bold: bool, base_font_size: float = 11.0) -> Optional[int]:
        """
        Detects heading level H1-H4 based on relative font size and boldness.
        """
        ratio = font_size / max(base_font_size, 8.0)
        if ratio >= 1.6:
            return 1
        elif ratio >= 1.3:
            return 2
        elif ratio >= 1.15 and is_bold:
            return 3
        elif is_bold and ratio >= 1.0:
            return 4
        return None

    @classmethod
    def build_canonical_document(
        cls,
        project_id: str,
        filename: str,
        pages_data: List[Dict[str, Any]],
        assets_data: List[Dict[str, Any]],
        source_toc: List[Dict[str, Any]] = None
    ) -> CanonicalDocument:
        """
        Reconstructs the hierarchical Canonical Document Model from extracted pages.
        """
        # Step 1: Filter repeated headers/footers
        repeated_hf = HeaderFooterFilter.identify_repeated_headers_footers(pages_data)

        # Calculate base font size mode
        font_sizes = []
        for p in pages_data:
            for b in p.get("blocks", []):
                if b.get("type") == "text" and "font_size" in b:
                    font_sizes.append(b["font_size"])
        base_font_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 11.0

        chapters: List[Chapter] = []
        current_chapter_nodes: List[DocumentNode] = []
        current_chapter_title = "Introduction"
        current_chapter_num = ""
        current_chapter_pages = []
        chapter_idx = 1
        node_counter = 1

        def finalize_chapter():
            nonlocal current_chapter_nodes, current_chapter_title, current_chapter_num, current_chapter_pages, chapter_idx
            if current_chapter_nodes or not chapters:
                ch_id = f"chapter_{chapter_idx:04d}"
                chapters.append(Chapter(
                    id=ch_id,
                    number=current_chapter_num if current_chapter_num else str(chapter_idx),
                    title=current_chapter_title,
                    level=1,
                    source_pages=list(set(current_chapter_pages)),
                    order_index=chapter_idx - 1,
                    nodes=current_chapter_nodes
                ))
                chapter_idx += 1
                current_chapter_nodes = []
                current_chapter_pages = []

        for p_data in pages_data:
            page_num = p_data.get("page_number", 1)
            page_height = p_data.get("height", 842.0)
            page_width = p_data.get("width", 595.0)
            raw_blocks = p_data.get("blocks", [])

            # Filter headers and footers
            filtered_blocks = HeaderFooterFilter.filter_blocks(raw_blocks, repeated_hf, page_height)
            
            # Sort reading order
            sorted_blocks = ParagraphRepair.sort_reading_order(filtered_blocks, page_width)

            for b in sorted_blocks:
                b_type = b.get("type", "text")
                b_id = b.get("block_id", 0)
                bbox = b.get("bbox", [0, 0, 0, 0])
                confidence = b.get("confidence", 1.0)
                source_mapping = SourceMapping(
                    source_document=filename,
                    source_page_start=page_num,
                    source_page_end=page_num,
                    source_block_ids=[b_id],
                    bounding_box=bbox
                )

                if b_type == "image":
                    asset_id = b.get("asset_id", "")
                    node_id = f"image_{chapter_idx:04d}_{node_counter:05d}"
                    current_chapter_nodes.append(DocumentNode(
                        id=node_id,
                        type=NodeType.IMAGE,
                        content=f"[Image Asset: {asset_id}]",
                        source_mapping=source_mapping,
                        metadata=NodeMetadata(image_asset_id=asset_id, confidence=confidence),
                        order_index=node_counter,
                    ))
                    node_counter += 1
                    current_chapter_pages.append(page_num)
                    continue

                # Text blocks
                raw_text = b.get("text", "").strip()
                if not raw_text:
                    continue

                # Repair hyphenation and broken lines
                cleaned_text = ParagraphRepair.repair_hyphenation(raw_text)
                font_size = b.get("font_size", base_font_size)
                is_bold = b.get("is_bold", False)
                is_italic = b.get("is_italic", False)
                font_name = b.get("font_name", "")

                # Check if Chapter Start
                chapter_match = cls.detect_chapter_heading(cleaned_text, font_size, is_bold, base_font_size)
                if chapter_match:
                    # Finalize previous chapter if it has content
                    if current_chapter_nodes:
                        finalize_chapter()
                    current_chapter_num, current_chapter_title, _ = chapter_match
                    current_chapter_pages.append(page_num)
                    
                    node_id = f"heading_{chapter_idx:04d}_{node_counter:05d}"
                    current_chapter_nodes.append(DocumentNode(
                        id=node_id,
                        type=NodeType.HEADING,
                        content=cleaned_text,
                        source_mapping=source_mapping,
                        metadata=NodeMetadata(
                            heading_level=1,
                            font_size=font_size,
                            font_weight="bold" if is_bold else "normal",
                            is_bold=is_bold,
                            confidence=confidence
                        ),
                        order_index=node_counter,
                    ))
                    node_counter += 1
                    continue

                # Check if Subheading (H2-H4)
                h_lvl = cls.detect_heading_level(font_size, is_bold, base_font_size)
                if h_lvl and len(cleaned_text.split()) <= 15:
                    node_id = f"heading_{chapter_idx:04d}_{node_counter:05d}"
                    current_chapter_nodes.append(DocumentNode(
                        id=node_id,
                        type=NodeType.HEADING,
                        content=cleaned_text,
                        source_mapping=source_mapping,
                        metadata=NodeMetadata(
                            heading_level=h_lvl,
                            font_size=font_size,
                            font_weight="bold" if is_bold else "normal",
                            is_bold=is_bold,
                            confidence=confidence
                        ),
                        order_index=node_counter,
                    ))
                    node_counter += 1
                    current_chapter_pages.append(page_num)
                    continue

                # Check if Footnote
                fn_match = cls.FOOTNOTE_PATTERN.match(cleaned_text)
                if fn_match and (bbox[1] > page_height * 0.75 or font_size < base_font_size * 0.9):
                    fn_num = fn_match.group(1) or fn_match.group(2) or "*"
                    node_id = f"footnote_{chapter_idx:04d}_{node_counter:05d}"
                    current_chapter_nodes.append(DocumentNode(
                        id=node_id,
                        type=NodeType.FOOTNOTE,
                        content=cleaned_text,
                        source_mapping=source_mapping,
                        metadata=NodeMetadata(
                            footnote_number=fn_num,
                            font_size=font_size,
                            confidence=confidence
                        ),
                        order_index=node_counter,
                    ))
                    node_counter += 1
                    current_chapter_pages.append(page_num)
                    continue

                # Default: Paragraph
                node_id = f"paragraph_{chapter_idx:04d}_{node_counter:05d}"
                current_chapter_nodes.append(DocumentNode(
                    id=node_id,
                    type=NodeType.PARAGRAPH,
                    content=cleaned_text,
                    source_mapping=source_mapping,
                    metadata=NodeMetadata(
                        font_size=font_size,
                        font_name=font_name,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        confidence=confidence
                    ),
                    order_index=node_counter,
                ))
                node_counter += 1
                current_chapter_pages.append(page_num)

        # Finalize last chapter
        finalize_chapter()

        # Build assets
        assets = [
            Asset(
                id=a["id"],
                original_path=a["original_path"],
                source_page=a.get("source_page", 1),
                width=a.get("width"),
                height=a.get("height"),
                mime_type=a.get("mime_type", "image/png")
            ) for a in assets_data
        ]

        total_words = sum(len(n.content.split()) for ch in chapters for n in ch.nodes)

        return CanonicalDocument(
            id=project_id,
            metadata=DocumentMetadata(
                title=filename.replace(".pdf", "").replace(".epub", "").replace("_", " ").title(),
                total_pages=len(pages_data),
                total_words=total_words,
                total_chapters=len(chapters),
                total_nodes=sum(len(ch.nodes) for ch in chapters),
            ),
            chapters=chapters,
            assets=assets,
        )
