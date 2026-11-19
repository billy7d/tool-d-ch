"""Nối lại đoạn văn bị cắt đôi giữa câu do sang trang.

PDF cho mỗi trang một khối chữ riêng, nên đoạn vắt qua hai trang thành hai node: "...this pain" | "seems relatively
unimportant...". Dịch riêng từng nửa thì model phải đoán phần câu còn thiếu (mất phủ định, sót động từ), và critic
chấm một câu cụt. Sách thật 842 đoạn có 172 chỗ như vậy.

Chỉ nối khi cả ba điều cùng đúng: hai đoạn văn liền nhau nằm ở hai trang kề nhau, đoạn trước chưa kết câu, và đoạn
sau mở đầu như phần giữa câu. Thà bỏ sót một chỗ ngắt còn hơn dính hai đoạn vốn tách rời.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from app.models.canonical import DocumentNode

# Dấu kết câu, có thể theo sau bởi ngoặc đóng, dấu chú thích hoặc số chú thích ("power.*", "grow.”12").
_SENTENCE_END = re.compile(r"[.!?:;…][\"'’”)\]*†]*\d{0,3}$")
_OPENING_QUOTES = "“\"‘"
_DASHES = "—–"
# Đoạn sau mở bằng ngoặc kép thì chỉ nối khi đoạn trước đủ dài để chắc là thân bài, không phải dòng đề từ hay tiêu đề.
_MIN_BODY_CHARS_BEFORE_QUOTE = 200


@dataclass
class Fragment:
    """Một node theo thứ tự đọc, đủ thông tin để quyết định nối."""
    kind: str  # "paragraph", "footnote" hoặc loại khác
    text: str
    page_start: int
    page_end: int
    joinable: bool = True


def continues_sentence(previous: str, following: str) -> bool:
    """Đoạn sau có phải phần còn lại của câu đang dở ở cuối đoạn trước không."""
    previous, following = previous.rstrip(), following.lstrip()
    if not previous or not following or _SENTENCE_END.search(previous):
        return False
    first, last = following[0], previous[-1]
    if first.islower() or first in _DASHES:
        return True
    if first in _OPENING_QUOTES:
        return len(previous) >= _MIN_BODY_CHARS_BEFORE_QUOTE and (last == "," or last.islower())
    return False


def join_text(previous: str, following: str) -> str:
    previous, following = previous.rstrip(), following.lstrip()
    # Từ có gạch nối bị cắt ở cuối trang ("self-" | "discipline"): giữ gạch nối, không thêm khoảng trắng.
    if len(previous) >= 2 and previous[-1] == "-" and previous[-2].isalpha() and following[:1].islower():
        return previous + following
    return f"{previous} {following}"


def find_joins(fragments: Sequence[Fragment]) -> List[Tuple[int, int]]:
    """Các cặp (vị trí đoạn giữ lại, vị trí đoạn bị gộp vào). Một đoạn vắt qua ba trang cho hai cặp cùng đoạn giữ."""
    joins: List[Tuple[int, int]] = []
    keeper: Optional[int] = None
    text, page_end = "", 0
    for index, fragment in enumerate(fragments):
        if keeper is not None:
            if fragment.kind == "footnote" and fragment.page_start == page_end:
                continue  # chú thích cuối trang nằm chen giữa hai nửa của đoạn
            if (
                fragment.kind == "paragraph" and fragment.joinable
                and fragment.page_start == page_end + 1 and continues_sentence(text, fragment.text)
            ):
                joins.append((keeper, index))
                text, page_end = join_text(text, fragment.text), fragment.page_end
                continue
        if fragment.kind == "paragraph" and fragment.joinable:
            keeper, text, page_end = index, fragment.text, fragment.page_end
        else:
            keeper = None
    return joins


def join_chapter_nodes(nodes: List[DocumentNode]) -> List[DocumentNode]:
    """Trả về danh sách node của một chương sau khi nối các đoạn bị ngắt trang."""
    fragments = [
        Fragment(node.type.value, node.content, node.source_mapping.source_page_start, node.source_mapping.source_page_end)
        for node in nodes
    ]
    absorbed = set()
    for keeper_index, absorbed_index in find_joins(fragments):
        keeper, extra = nodes[keeper_index], nodes[absorbed_index]
        keeper.content = join_text(keeper.content, extra.content)
        keeper.source_mapping.source_page_end = extra.source_mapping.source_page_end
        keeper.source_mapping.source_block_ids.extend(extra.source_mapping.source_block_ids)
        keeper.metadata.confidence = min(keeper.metadata.confidence, extra.metadata.confidence)
        absorbed.add(absorbed_index)
    return [node for index, node in enumerate(nodes) if index not in absorbed]


def join_pending_nodes(db, project_id: str) -> int:
    """Nối các đoạn bị ngắt trang của dự án đã nhập từ trước, chỉ khi cả hai nửa chưa hề được dịch.

    Trả về số đoạn đã gộp. Không tự commit.
    """
    from app.db.models import ChapterModel, NodeModel

    merged = 0
    chapters = db.query(ChapterModel).filter(ChapterModel.project_id == project_id).order_by(ChapterModel.order_index).all()
    for chapter in chapters:
        rows = db.query(NodeModel).filter(NodeModel.chapter_id == chapter.id).order_by(NodeModel.order_index).all()
        fragments = []
        for row in rows:
            mapping = row.source_mapping or {}
            untouched = (
                row.status == "PENDING" and not row.translated_content and (row.version or 1) == 1
                and not (row.node_metadata or {}).get("draft_translation")
            )
            fragments.append(Fragment(
                row.node_type, row.content or "",
                int(mapping.get("source_page_start") or 0), int(mapping.get("source_page_end") or 0),
                # Thiếu số trang (dự án không nhập từ file có trang) thì không có căn cứ để nối.
                joinable=untouched and bool(mapping.get("source_page_start")),
            ))
        for keeper_index, absorbed_index in find_joins(fragments):
            keeper, extra = rows[keeper_index], rows[absorbed_index]
            keeper_mapping, extra_mapping = dict(keeper.source_mapping or {}), extra.source_mapping or {}
            keeper.content = join_text(keeper.content, extra.content)
            keeper_mapping["source_page_end"] = extra_mapping.get("source_page_end", keeper_mapping.get("source_page_end"))
            keeper_mapping["source_block_ids"] = list(keeper_mapping.get("source_block_ids") or []) + list(extra_mapping.get("source_block_ids") or [])
            keeper.source_mapping = keeper_mapping
            db.delete(extra)
            merged += 1
    return merged


__all__ = ["Fragment", "continues_sentence", "join_text", "find_joins", "join_chapter_nodes", "join_pending_nodes"]
