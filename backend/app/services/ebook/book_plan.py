"""Biến CanonicalDocument thành "bản dựng sách" chung cho bộ ghi EPUB và MOBI.

Bản dựng quyết định những gì độc lập với định dạng: chia file (mỗi file mở trang mới trong KOReader),
mục lục lồng nhau, đưa chú thích chữ nhỏ về cuối chương, gộp chương chỉ có tiêu đề vào chương sau,
và chuyển chữ (bản dịch nếu có, không thì bản gốc) thành XHTML nội dòng an toàn.
"""
from __future__ import annotations

import html
import re
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.models.canonical import CanonicalDocument, Chapter, DocumentNode, NodeType
from app.services.ebook.text_cleanup import inline_markup, strip_markers, xml_safe

# Nhãn theo ngôn ngữ của sách.
_LABELS = {
    "vi": {"toc": "Mục lục", "cover": "Bìa", "start": "Bắt đầu đọc", "notes": "Chú thích", "front": "Phần đầu sách"},
    "en": {"toc": "Contents", "cover": "Cover", "start": "Start", "notes": "Notes", "front": "Front matter"},
}
_DEFAULT_CHAPTER_TITLES = {"introduction", "untitled chapter", ""}
# Ngưỡng chia file: file lớn làm máy đọc cũ mở chậm; ưu tiên chia ở tiêu đề.
SOFT_FILE_LIMIT = 120_000
HARD_FILE_LIMIT = 260_000


def labels_for(language: str) -> Dict[str, str]:
    return _LABELS.get((language or "en").split("-")[0].lower(), _LABELS["en"])


@dataclass
class PlanBlock:
    kind: str  # heading | para | image | list | table | code | quote | rule | footnotes
    html: str = ""
    text: str = ""
    level: int = 0
    anchor: str = ""
    classes: Tuple[str, ...] = ()
    image_key: str = ""
    items: List[str] = field(default_factory=list)
    ordered: bool = False
    rows: List[List[str]] = field(default_factory=list)

    def weight(self) -> int:
        return len(self.html) + sum(len(i) for i in self.items) + sum(len(c) for r in self.rows for c in r) + 40


@dataclass
class PlanFile:
    key: str
    blocks: List[PlanBlock] = field(default_factory=list)

    def weight(self) -> int:
        return sum(b.weight() for b in self.blocks)


@dataclass
class TocNode:
    title: str
    file_key: str
    anchor: str
    level: int
    children: List["TocNode"] = field(default_factory=list)


@dataclass
class PlanImage:
    key: str
    path: Optional[Path] = None
    data: Optional[bytes] = None

    def load(self) -> Optional[bytes]:
        if self.data is not None:
            return self.data
        try:
            return self.path.read_bytes() if self.path else None
        except OSError:
            return None


@dataclass
class BookStyle:
    paragraph_style: str = "indent"  # indent | block
    indent_em: float = 1.2
    justify: bool = True


@dataclass
class BookPlan:
    title: str
    author: str
    translator: str
    language: str
    identifier: str
    modified: str
    files: List[PlanFile]
    toc: List[TocNode]
    images: Dict[str, PlanImage]
    cover: Optional[bytes]
    style: BookStyle

    @property
    def labels(self) -> Dict[str, str]:
        return labels_for(self.language)

    def toc_depth(self) -> int:
        def depth(nodes: Sequence[TocNode]) -> int:
            return max((1 + depth(n.children) for n in nodes), default=0)

        return max(1, depth(self.toc))

    def flat_toc(self) -> List[TocNode]:
        out: List[TocNode] = []

        def walk(nodes: Sequence[TocNode]) -> None:
            for n in nodes:
                out.append(n)
                walk(n.children)

        walk(self.toc)
        return out


def _node_text(node: DocumentNode) -> str:
    return node.translated_content if (node.translated_content or "").strip() else (node.content or "")


def _is_translated(doc: CanonicalDocument) -> bool:
    return any((n.translated_content or "").strip() for ch in doc.chapters for n in ch.nodes)


def _layout(node: DocumentNode) -> dict:
    return (node.metadata.extra or {}).get("layout") or {}


# Tiêu đề chỉ là nhãn đánh số ("Chương 3", "PART II", "IV"): gộp với tiêu đề ngay sau thành một mục lục.
_LABEL_HEADING_RE = re.compile(
    r"^(?:chapter|part|section|book|volume|appendix|chương|phần|mục|quyển|tập|phụ lục|bài|hồi)\b[\s\w.\-]{0,12}$"
    r"|^[\dIVXLCDM]{1,6}[.)]?$",
    re.IGNORECASE,
)


def _is_label_heading(text: str) -> bool:
    return bool(_LABEL_HEADING_RE.match(text.strip())) and len(text.split()) <= 3


def _heading_join(first: str, second: str) -> str:
    if re.search(r"[.:;!?—–-]$", first):
        return f"{first} {second}"
    return f"{first}: {second}"


class _PlanBuilder:
    def __init__(self, doc: CanonicalDocument, style: BookStyle, language: str, book_title: str = "") -> None:
        self.doc = doc
        self.book_title = book_title
        self.style = style
        self.language = language
        self.files: List[PlanFile] = []
        self.toc_entries: List[Tuple[int, str, str, str]] = []  # (level, title, file_key, anchor)
        self.images: Dict[str, PlanImage] = {}
        self.asset_paths = {a.id: a for a in doc.assets}
        self.anchor_count = 0
        self.body_size = self._body_font_size()

    def _body_font_size(self) -> Optional[float]:
        sizes = [
            n.metadata.font_size for ch in self.doc.chapters for n in ch.nodes
            if n.type == NodeType.PARAGRAPH and n.metadata.font_size
        ]
        return statistics.median(sizes) if sizes else None

    def _new_anchor(self) -> str:
        self.anchor_count += 1
        return f"h{self.anchor_count:04d}"

    def _new_file(self) -> PlanFile:
        plan_file = PlanFile(key=f"part{len(self.files) + 1:04d}")
        self.files.append(plan_file)
        return plan_file

    # ---- Chương ------------------------------------------------------------------------------

    def _merged_chapters(self) -> List[Tuple[Chapter, List[DocumentNode]]]:
        """Chương chỉ có tiêu đề (trang "PHẦN I" đứng riêng) gộp vào chương sau để không sinh trang trống."""
        result: List[Tuple[Chapter, List[DocumentNode]]] = []
        pending: List[DocumentNode] = []
        for chapter in sorted(self.doc.chapters, key=lambda c: c.order_index):
            nodes = pending + list(chapter.nodes)
            pending = []
            if nodes and all(n.type == NodeType.HEADING for n in nodes):
                pending = nodes
                continue
            if nodes:
                result.append((chapter, nodes))
        if pending:
            if result:
                result[-1][1].extend(pending)
            elif self.doc.chapters:
                result.append((self.doc.chapters[0], pending))
        return result

    def _is_relocatable_footnote(self, node: DocumentNode) -> bool:
        if node.type != NodeType.FOOTNOTE or not self.body_size or not node.metadata.font_size:
            return False
        return node.metadata.font_size < 0.92 * self.body_size

    # ---- Khối --------------------------------------------------------------------------------

    def _paragraph_classes(self, node: DocumentNode, after_heading: bool) -> Tuple[str, ...]:
        layout = _layout(node)
        classes: List[str] = []
        align = layout.get("align")
        if layout.get("quote"):
            classes.append("quote")
            if self.style.paragraph_style == "indent" and not layout.get("indent"):
                classes.append("noindent")
            return tuple(classes)
        if layout.get("scene"):
            classes.append("scene")
        elif layout.get("verse"):
            classes.append("verse")
            if align == "center":
                classes.append("center")
        elif align in ("center", "right"):
            classes.append(align)
        elif self.style.paragraph_style == "indent":
            if layout:
                if not layout.get("indent"):
                    classes.append("noindent")
            elif after_heading:
                classes.append("noindent")
        return tuple(classes)

    def _image_block(self, node: DocumentNode) -> Optional[PlanBlock]:
        asset_id = node.metadata.image_asset_id or ""
        asset = self.asset_paths.get(asset_id)
        if not asset:
            return None
        path = Path(asset.derived_path or asset.original_path)
        if not path.exists():
            path = Path(asset.original_path)
        if not path.exists():
            return None
        key = f"img{len(self.images) + 1:04d}"
        for existing_key, image in self.images.items():
            if image.path == path:
                key = existing_key
                break
        else:
            self.images[key] = PlanImage(key=key, path=path)
        caption = (node.metadata.image_caption or "").strip()
        return PlanBlock(kind="image", image_key=key, html=inline_markup(caption) if caption else "",
                         text=strip_markers(caption))

    def _render_node(self, node: DocumentNode, after_heading: bool) -> Optional[PlanBlock]:
        text = _node_text(node)
        if node.type == NodeType.IMAGE:
            return self._image_block(node)
        if node.type == NodeType.HORIZONTAL_RULE:
            return PlanBlock(kind="rule")
        if node.type == NodeType.PAGE_BREAK_HINT:
            return None
        if node.type == NodeType.CODE_BLOCK:
            if not text.strip():
                return None
            return PlanBlock(kind="code", html=html.escape(xml_safe(strip_markers(text)), quote=False))
        if node.type == NodeType.TABLE and node.metadata.table_rows:
            rows = [[inline_markup(str(cell)) for cell in row] for row in node.metadata.table_rows]
            return PlanBlock(kind="table", rows=rows)
        if not strip_markers(text).strip():
            return None
        markup = inline_markup(text)
        if node.type == NodeType.QUOTE:
            return PlanBlock(kind="quote", html=markup, text=strip_markers(text))
        if node.type == NodeType.FOOTNOTE:
            return PlanBlock(kind="para", html=markup, text=strip_markers(text), classes=("footnote",))
        if node.type in (NodeType.LIST_ITEM,) or node.metadata.list_type:
            return PlanBlock(kind="list", items=[markup], ordered=node.metadata.list_type == "numbered")
        return PlanBlock(kind="para", html=markup, text=strip_markers(text),
                         classes=self._paragraph_classes(node, after_heading))

    # ---- Dựng --------------------------------------------------------------------------------

    def _title_page_headings(self, chapters) -> set:
        """Chuỗi >= 3 tiêu đề liền nhau ở đầu sách là trang tên sách: mục lục chỉ giữ một mục."""
        run = []
        for _, nodes in chapters:
            for node in nodes:
                if node.type == NodeType.HEADING:
                    run.append(node.id)
                elif strip_markers(_node_text(node)).strip() or node.type == NodeType.IMAGE:
                    return set(run) if len(run) >= 3 else set()
        return set(run) if len(run) >= 3 else set()

    def build(self) -> None:
        chapters = self._merged_chapters()
        title_page = self._title_page_headings(chapters)
        title_page_listed = False
        for chapter, nodes in chapters:
            current = self._new_file()
            footnotes: List[str] = []
            starts_with_heading = any(n.type == NodeType.HEADING for n in nodes[:3])
            title = (chapter.translated_title or chapter.title or "").strip()
            if not starts_with_heading and title.lower() not in _DEFAULT_CHAPTER_TITLES:
                # Chương có tên nhưng thân không mở bằng tiêu đề: thêm tiêu đề để có chỗ cho mục lục.
                anchor = self._new_anchor()
                current.blocks.append(PlanBlock(kind="heading", level=1, anchor=anchor, html=inline_markup(title),
                                                text=strip_markers(title), classes=("center",)))
                self.toc_entries.append((1, strip_markers(title), current.key, anchor))
            elif not starts_with_heading and nodes and not self.toc_entries and len(self.doc.chapters) > 1:
                self.toc_entries.append((1, labels_for(self.language)["front"], current.key, ""))

            after_heading = False
            previous_was_heading = False
            for node in nodes:
                if self._is_relocatable_footnote(node):
                    text = _node_text(node)
                    if strip_markers(text).strip():
                        footnotes.append(inline_markup(text))
                    continue

                if node.type == NodeType.HEADING:
                    heading_text = strip_markers(_node_text(node)).strip()
                    if not heading_text:
                        continue
                    level = max(1, min(4, node.metadata.heading_level or 2))
                    starts_group = not previous_was_heading
                    if starts_group and current.blocks and any(b.kind != "heading" for b in current.blocks):
                        weight = current.weight()
                        if level == 1 or weight >= SOFT_FILE_LIMIT:
                            current = self._new_file()
                    anchor = self._new_anchor()
                    layout = _layout(node)
                    align = layout.get("align")
                    classes: Tuple[str, ...] = ()
                    if align in ("center", "right"):
                        classes = (align,)
                    elif layout and level == 1:
                        classes = ("left",)
                    current.blocks.append(PlanBlock(kind="heading", level=level, anchor=anchor,
                                                    html=inline_markup(_node_text(node)), text=heading_text,
                                                    classes=classes))
                    if node.id in title_page:
                        if not title_page_listed:
                            self.toc_entries.append((0, self.book_title or heading_text, current.key, anchor))
                            title_page_listed = True
                    elif (
                        previous_was_heading and self.toc_entries and self.toc_entries[-1][2] == current.key
                        and _is_label_heading(self.toc_entries[-1][1])
                    ):
                        lvl, prev_title, key, prev_anchor = self.toc_entries[-1]
                        self.toc_entries[-1] = (min(lvl, level), _heading_join(prev_title, heading_text), key, prev_anchor)
                    else:
                        self.toc_entries.append((level, heading_text, current.key, anchor))
                    previous_was_heading = True
                    after_heading = True
                    continue

                block = self._render_node(node, after_heading)
                previous_was_heading = False
                if block is None:
                    continue
                after_heading = False
                if current.weight() >= HARD_FILE_LIMIT and block.kind != "list":
                    current = self._new_file()
                if block.kind == "list" and current.blocks and current.blocks[-1].kind == "list" \
                        and current.blocks[-1].ordered == block.ordered:
                    current.blocks[-1].items.extend(block.items)
                else:
                    current.blocks.append(block)

            if footnotes:
                current.blocks.append(PlanBlock(kind="footnotes", items=footnotes))
        self.files = [f for f in self.files if f.blocks]
        if not self.files:
            self.files.append(PlanFile(key="part0001", blocks=[PlanBlock(kind="para", html="", classes=("noindent",))]))

    def toc(self, title: str) -> List[TocNode]:
        entries = [e for e in self.toc_entries if e[1].strip()]
        valid_keys = {f.key for f in self.files}
        entries = [e for e in entries if e[2] in valid_keys]
        if not entries:
            return [TocNode(title=title, file_key=self.files[0].key, anchor="", level=1)]
        used_levels = sorted({e[0] for e in entries if e[0] > 0})
        depth_of = {lvl: min(index + 1, 3) for index, lvl in enumerate(used_levels)}
        roots: List[TocNode] = []
        stack: List[TocNode] = []
        for index, (level, text, key, anchor) in enumerate(entries):
            if level <= 0:
                # Mục "trang tên sách": đứng ngang hàng mục kế tiếp, không làm mục cha của nó.
                following = next((e[0] for e in entries[index + 1:] if e[0] > 0), None)
                depth = depth_of.get(following, 1) if following else 1
            else:
                depth = depth_of[level]
            node = TocNode(title=text, file_key=key, anchor=anchor, level=depth)
            while stack and stack[-1].level >= depth:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
        return roots


def build_book_plan(
    doc: CanonicalDocument,
    language: Optional[str] = None,
    style: Optional[BookStyle] = None,
    cover: Optional[bytes] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    translator: Optional[str] = None,
) -> BookPlan:
    translated = _is_translated(doc)
    lang = (language or (doc.metadata.target_language if translated else doc.metadata.language) or "en").strip()
    style = style or BookStyle()
    book_title = (title or doc.metadata.title or "Sách").strip()
    builder = _PlanBuilder(doc, style, lang, book_title=book_title)
    builder.build()
    return BookPlan(
        title=book_title,
        author=(author if author is not None else (doc.metadata.author or "")).strip(),
        translator=((translator if translator is not None else doc.metadata.translator) or "").strip() if translated else "",
        language=lang,
        identifier=f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, 'tool-dich:' + (doc.id or book_title) + ':' + book_title)}",
        modified=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        files=builder.files,
        toc=builder.toc(book_title),
        images=builder.images,
        cover=cover,
        style=style,
    )
