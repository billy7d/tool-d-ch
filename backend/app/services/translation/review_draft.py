"""Bản nháp bị chặn của một đoạn, lưu trong node_metadata["draft_translation"].

Khi gate hoặc critic chặn, hệ thống không lưu candidate làm bản dịch chính thức (fail-closed).
Trước đây candidate bị bỏ hẳn nên ở Bước 6 người duyệt thấy ô trống và phải dịch lại từ đầu,
kể cả khi bản của model chỉ sai một chi tiết. Bản nháp không bao giờ vào translated_content hay TM.
"""
from datetime import datetime
from typing import Any, Iterable, Optional

DRAFT_KEY = "draft_translation"
MAX_REASONS = 8


def save_review_draft(node: Any, text: Optional[str], reasons: Iterable[str], stage: str) -> None:
    if node is None or not str(text or "").strip():
        return
    metadata = dict(getattr(node, "node_metadata", None) or {})
    metadata[DRAFT_KEY] = {
        "text": str(text).strip(),
        "reasons": sorted({str(reason) for reason in reasons if reason})[:MAX_REASONS],
        "stage": stage,
        "saved_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    # Gán dict mới để SQLAlchemy nhận ra thay đổi của cột JSON.
    node.node_metadata = metadata


def clear_review_draft(node: Any) -> None:
    metadata = getattr(node, "node_metadata", None) or {}
    if node is not None and DRAFT_KEY in metadata:
        node.node_metadata = {key: value for key, value in metadata.items() if key != DRAFT_KEY}
