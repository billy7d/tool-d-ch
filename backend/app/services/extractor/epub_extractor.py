import copy
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup, Tag

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None
    epub = None

_HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")
# Khối chứa chữ của sách. Dịch thử Pro Git: bản cũ chỉ đọc h1-h6 và p nên mất 26% chữ (danh sách, khối code,
# chú thích hình); Origin of Species mất 235 mục định nghĩa, Wealth of Nations mất các bảng.
_BLOCKS = _HEADINGS + ("p", "li", "pre", "figcaption", "dt", "dd", "blockquote", "tr", "img")
# Khối con đã được đọc riêng: khối cha chỉ lấy phần chữ còn lại để không lặp nội dung.
_NESTED = _HEADINGS + ("p", "ul", "ol", "dl", "pre", "table", "blockquote", "figure", "li", "dt", "dd")
_CONTAINERS = ("li", "figcaption", "dt", "dd", "blockquote")
_INLINE_CODE = ("code", "kbd", "samp", "tt")
# Ký tự vô hình của sách điện tử ("laws﻿—Unity"): model chép lại vào bản dịch.
_INVISIBLE = re.compile("[" + "".join(map(chr, (0xFEFF, 0x2060, 0x200B, 0x00AD))) + "]")


def _clean(text: str) -> str:
    return " ".join(_INVISIBLE.sub("", text or "").split())


def _epub_type(element: Tag) -> str:
    return str(element.get("epub:type") or element.get("data-epub-type") or "")


def _prepare(soup: BeautifulSoup) -> None:
    """Giữ dấu hiệu mà get_text làm mất: code trong dòng và số chú thích."""
    for element in soup.find_all(_INLINE_CODE):
        if element.find_parent("pre"):
            continue
        text = element.get_text()
        # Dịch thử Pro Git: mất dấu code thì "git apply" bị dịch thành "git áp dụng".
        element.replace_with(f"`{text}`" if text.strip() and "`" not in text else text)
    for link in soup.find_all("a"):
        kind = _epub_type(link)
        if "backlink" in kind:
            link.decompose()  # "↩︎" ở cuối chú thích
        elif "noteref" in kind:
            # Số chú thích dính vào chữ ("twenty percent.697") thành số liệu lạ trong bản dịch.
            link.replace_with(f" [{link.get_text().strip()}]")


def _own_text(element: Tag) -> str:
    """Chữ của khối, không tính các khối con đã đọc riêng."""
    clone = copy.copy(element)
    for nested in clone.find_all(_NESTED):
        nested.decompose()
    return _clean(clone.get_text(" "))


def _inside(element: Tag, names) -> bool:
    return element.find_parent(names) is not None


class EPUBExtractor:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.assets_dir = project_dir / "assets" / "original"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _blocks(soup: BeautifulSoup) -> List[Dict[str, Any]]:
        body = soup.body or soup
        blocks: List[Dict[str, Any]] = []

        def add(kind: str, text: str, **extra: Any) -> None:
            if kind != "image" and not text:
                return
            block = {"block_id": len(blocks), "type": kind, "confidence": 1.0, "structured": True, **extra}
            if kind != "image":
                block.update({"text": text, "font_size": extra.get("font_size", 11.0), "is_bold": kind == "heading"})
            blocks.append(block)

        for element in body.find_all(_BLOCKS):
            tag = element.name.lower()
            if tag != "pre" and _inside(element, "pre"):
                continue
            if tag in _HEADINGS:
                level = int(tag[1])
                add("heading", _clean(element.get_text(" ")), heading_level=level, font_size=18.0 - level * 2)
            elif tag == "p":
                if _inside(element, _HEADINGS):
                    continue
                kind = _epub_type(element)
                if element.find_parent("hgroup") or "title" in kind.split() or "subtitle" in kind:
                    # Standard Ebooks: <hgroup><h2>VII</h2><p epub:type="title">Miscellaneous Objections…</p>.
                    parent_heading = element.find_parent("hgroup")
                    first = parent_heading.find(_HEADINGS) if parent_heading else None
                    level = min(6, int(first.name[1]) + 1) if first else 3
                    add("heading", _clean(element.get_text(" ")), heading_level=level, font_size=18.0 - level * 2)
                else:
                    add("text", _clean(element.get_text(" ")))
            elif tag == "pre":
                if _inside(element, "pre"):
                    continue
                code = element.find("code")
                classes = " ".join((code or element).get("class") or [])
                language = re.search(r"language-([\w+-]+)", classes)
                # Lấy chữ của <code> bên trong: khoảng trắng giữa <pre> và <code> là định dạng HTML, không phải code.
                text = _INVISIBLE.sub("", (code or element).get_text()).strip("\n")
                add("code", text, code_language=language.group(1) if language else None)
            elif tag == "tr":
                # Ô có <p> đã được đọc như đoạn văn; bảng chỉ có chữ trong ô thì giữ cả hàng thành một khối.
                if element.find("p"):
                    continue
                cells = [_clean(cell.get_text(" ")) for cell in element.find_all(["td", "th"], recursive=False)]
                add("text", " | ".join(cell for cell in cells if cell))
            elif tag in _CONTAINERS:
                add("text", _own_text(element), list_item=tag == "li")
            elif tag == "img":
                add("image", "", asset_id=str(uuid.uuid4()))
        return blocks

    def extract_document(self, epub_path: Path) -> Dict[str, Any]:
        if epub is None:
            raise ImportError("ebooklib is required for EPUB extraction.")

        book = epub.read_epub(str(epub_path))
        pages_data = []
        assets_data = []
        toc_entries = []

        # Đọc theo spine (thứ tự đọc của sách). get_items() trả theo manifest, thường là thứ tự tên file: Sherlock
        # Holmes từng ra "A Case of Identity" trước "A Scandal in Bohemia" và trang Colophon nằm giữa các truyện.
        documents = []
        for idref, *_ in book.spine:
            spine_item = book.get_item_with_id(idref)
            if spine_item is not None and spine_item.get_type() == ebooklib.ITEM_DOCUMENT and spine_item not in documents:
                documents.append(spine_item)
        if not documents:
            documents = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
        images = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_IMAGE]

        page_num = 1
        for item in documents + images:
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                # get_content() của ebooklib dựng lại tài liệu với <head/> rỗng; tiêu đề nằm trong nội dung gốc.
                title_tag: Optional[Tag] = BeautifulSoup(getattr(item, "content", b"") or b"", "html.parser").find("title")
                _prepare(soup)
                page_blocks = self._blocks(soup)
                if page_blocks:
                    pages_data.append({
                        "page_number": page_num,
                        "page_type": "TEXT_PAGE",
                        "title": _clean(title_tag.get_text()) if title_tag else "",
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
