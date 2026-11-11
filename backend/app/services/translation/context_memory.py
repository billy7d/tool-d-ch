import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.db.models import NodeModel


CHAPTER_MEMORY_VERSION = "chapter-memory-v3-structured"
CHAPTER_MEMORY_PROMPT_VERSION = "chapter-memory-prompt-v4-schema"
CHAPTER_MEMORY_FIELDS = {
    "summary", "entities", "key_concepts", "tone", "pronoun_notes",
    "terminology", "important_facts", "style_notes",
}
# Đủ cho khoảng một đoạn trung bình hoặc vài câu thoại. So với 700: tỷ lệ đạt gate không đổi (30/40) mà
# nhanh hơn ~10% vì phần ngữ cảnh này phải đọc lại ở mỗi lần gọi.
ROLLING_CONTEXT_MAX_TOKENS = 400
# Đoạn trước rất dài chỉ giữ phần cuối: vài câu sát đoạn đang dịch mới có ích cho đại từ và mạch văn, còn
# cả đoạn 2.700 ký tự làm mọi lệnh gọi cho đoạn sau tốn thêm nhiều giây đọc lại.
ROLLING_ITEM_MAX_CHARS = 600
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…”\"])\s+")
CHAPTER_MEMORY_LIST_LIMITS = {
    "entities": 20,
    "key_concepts": 15,
    "pronoun_notes": 10,
    "terminology": 30,
    "important_facts": 15,
    "style_notes": 10,
}


_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
# Chữ viết hoa đầu câu như "This", "But", "Give" không phải tên riêng; đo thật chúng lọt vào prompt
# dưới dạng "But" -> "But", tức bảo model giữ nguyên từ tiếng Anh.
_NOT_A_NAME = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "give", "he", "her", "his", "how", "i", "if", "in",
    "it", "its", "life", "love", "my", "no", "not", "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "then", "there", "these", "they", "this", "those", "to", "we", "what", "when", "where", "which", "while",
    "who", "why", "with", "yes", "you", "your",
}


def _note_text(item: Any) -> Any:
    """Model nhỏ hay trả ghi chú dạng {"source": "I", "preferred": "tôi"}; đo thật đây là lý do cả Chapter Memory bị bỏ."""
    if isinstance(item, dict) and {"source", "preferred"} <= set(item):
        return f'{item["source"]} -> {item["preferred"]}'
    return item


def _without_cjk(items: Any) -> Any:
    """Bỏ mục lẫn chữ Hán (đo thật: "postulation of a God" -> "tạo hóa một神的存在…") để không nhiễm vào prompt dịch."""
    if not isinstance(items, list):
        return items
    return [item for item in items if not _CJK.search(json.dumps(item, ensure_ascii=False))]


def validate_chapter_memory_response(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or set(value) != CHAPTER_MEMORY_FIELDS:
        raise ValueError("Chapter Memory phải là object đúng schema.")
    value = dict(value)
    for field_name in CHAPTER_MEMORY_LIST_LIMITS:
        items = _without_cjk(value[field_name])
        if field_name not in {"entities", "terminology"} and isinstance(items, list):
            items = [_note_text(item) for item in items]
        value[field_name] = items
    if not isinstance(value["summary"], str) or len(value["summary"]) > 1800:
        raise ValueError("Chapter Memory summary không hợp lệ.")
    if not isinstance(value["tone"], str) or len(value["tone"]) > 500:
        raise ValueError("Chapter Memory tone không hợp lệ.")
    normalized: Dict[str, Any] = {"summary": value["summary"].strip(), "tone": value["tone"].strip()}
    for field_name, limit in CHAPTER_MEMORY_LIST_LIMITS.items():
        items = value[field_name]
        if not isinstance(items, list) or len(items) > limit:
            raise ValueError(f"Chapter Memory {field_name} không hợp lệ.")
        if field_name in {"entities", "terminology"}:
            if not all(
                isinstance(item, dict)
                and set(item) == {"source", "preferred"}
                and isinstance(item["source"], str)
                and isinstance(item["preferred"], str)
                and len(item["source"]) <= 200
                and len(item["preferred"]) <= 200
                for item in items
            ):
                raise ValueError(f"Chapter Memory {field_name} phải gồm các cặp source/preferred.")
            normalized[field_name] = [
                {"source": item["source"].strip(), "preferred": item["preferred"].strip()}
                for item in items if item["source"].strip()
            ]
        else:
            if not all(isinstance(item, str) and len(item) <= 500 for item in items):
                raise ValueError(f"Chapter Memory {field_name} phải là danh sách chuỗi hợp lệ.")
            normalized[field_name] = [item.strip() for item in items if item.strip()]
    return normalized


@dataclass
class ChapterMemory:
    summary: str = ""
    entities: List[Dict[str, str]] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    tone: str = ""
    pronoun_notes: List[str] = field(default_factory=list)
    terminology: List[Dict[str, str]] = field(default_factory=list)
    important_facts: List[str] = field(default_factory=list)
    style_notes: List[str] = field(default_factory=list)
    version: str = CHAPTER_MEMORY_VERSION
    generated_at: str = ""
    source_hash: str = ""
    prompt_version: str = CHAPTER_MEMORY_VERSION
    glossary_hash: str = ""
    document_type: str = "GENERAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Chỉ phần có ý nghĩa với người dịch; hash và metadata cache chỉ tốn token (~100 token mỗi request)."""
        return {name: getattr(self, name) for name in sorted(CHAPTER_MEMORY_FIELDS) if getattr(self, name)}

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ChapterMemory":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value[key] for key in allowed if key in value})


@dataclass(frozen=True)
class BilingualContextItem:
    node_id: str
    source: str
    translation: str


def _ending_of(text: str) -> str:
    """Phần cuối của đoạn, bắt đầu từ đầu một câu để model không gặp câu bị cắt dở."""
    if len(text) <= ROLLING_ITEM_MAX_CHARS:
        return text
    ending = text[-ROLLING_ITEM_MAX_CHARS:]
    boundary = _SENTENCE_BOUNDARY.search(ending)
    return ending[boundary.end():] if boundary else ending


class ChapterMemoryBuilder:
    @staticmethod
    def source_hash(nodes: List[Any]) -> str:
        text = "\n".join(str(getattr(node, "content", "") or "") for node in nodes)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def stratified_sample(nodes: List[Any], max_chars: int = 3000) -> str:
        usable = [node for node in nodes if str(getattr(node, "content", "") or "").strip()]
        if not usable:
            return ""
        indexes = set()
        ranges = ((0.0, 0.15), (0.2, 0.35), (0.4, 0.6), (0.65, 0.8), (0.85, 1.0))
        for start, end in ranges:
            indexes.add(min(len(usable) - 1, int((len(usable) - 1) * ((start + end) / 2))))
        for index, node in enumerate(usable):
            node_type = str(getattr(node, "node_type", getattr(node, "type", ""))).lower()
            text = str(getattr(node, "content", "") or "")
            if "heading" in node_type or len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)) >= 2:
                indexes.add(index)
        selected = [str(getattr(usable[index], "content", "")) for index in sorted(indexes)]
        quota = max(120, max_chars // max(1, len(selected)))
        return "\n\n".join(text[:quota] for text in selected)[:max_chars]

    @staticmethod
    def _entities(sample: str, glossary: Dict[str, str]) -> List[Dict[str, str]]:
        found: List[Dict[str, str]] = []
        seen = set()
        for source, preferred in glossary.items():
            if re.search(rf"(?i)(?<!\w){re.escape(source)}(?!\w)", sample):
                found.append({"source": source, "preferred": preferred})
                seen.add(source.lower())
        # Dấu cách thay vì \s: một cụm viết hoa không được nối qua hai dòng ("Psychotherapy\n\nThe Baby").
        for name in re.findall(r"\b[A-Z][A-Za-z.&'-]+(?: [A-Z][A-Za-z.&'-]+){0,3}\b", sample):
            if name.lower() not in seen and len(name) > 2 and name.lower() not in _NOT_A_NAME:
                found.append({"source": name, "preferred": name})
                seen.add(name.lower())
            if len(found) >= 20:
                break
        return found

    @classmethod
    def load_or_create(
        cls,
        chapter_id: str,
        chapter_title: str,
        nodes: List[Any],
        cache_dir: Path,
        glossary: Optional[Dict[str, str]] = None,
        provider: Optional[Any] = None,
        model_name: Optional[str] = None,
        document_type: str = "GENERAL",
    ) -> ChapterMemory:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"chapter_context_{chapter_id}.json"
        source_digest = cls.source_hash(nodes)
        glossary_signature = json.dumps(glossary or {}, ensure_ascii=False, sort_keys=True)
        glossary_digest = hashlib.sha256(glossary_signature.encode("utf-8")).hexdigest()
        normalized_document_type = (document_type or "GENERAL").upper()
        if cache_path.exists():
            try:
                cached = ChapterMemory.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
                if (
                    cached.source_hash == source_digest
                    and cached.glossary_hash == glossary_digest
                    and cached.document_type == normalized_document_type
                    and cached.version == CHAPTER_MEMORY_VERSION
                    and cached.prompt_version == CHAPTER_MEMORY_PROMPT_VERSION
                ):
                    return cached
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        sample = cls.stratified_sample(nodes)
        # Fallback không đưa nguyên văn nguồn vào context khi provider không tạo được summary.
        summary = (
            f"Chương '{chapter_title}' thuộc lĩnh vực {normalized_document_type}. "
            "Không có tóm tắt ngữ nghĩa đáng tin cậy từ provider."
            if sample else f"Chương '{chapter_title}' chưa có nội dung tóm tắt."
        )
        structured: Dict[str, Any] = {}
        if provider and sample:
            try:
                generated = provider.build_chapter_memory(
                    sample, chapter_title, normalized_document_type, model=model_name,
                )
                structured = validate_chapter_memory_response(generated)
                summary = structured["summary"] or summary
            except Exception:
                structured = {}
        memory = ChapterMemory(
            summary=summary,
            entities=structured.get("entities") or cls._entities(sample, glossary or {}),
            key_concepts=structured.get("key_concepts") or [],
            tone=str(structured.get("tone") or "Giữ nhất quán với document profile và phần đã dịch."),
            pronoun_notes=structured.get("pronoun_notes") or [],
            terminology=structured.get("terminology") or [{"source": key, "preferred": value} for key, value in (glossary or {}).items() if key.lower() in sample.lower()][:30],
            important_facts=structured.get("important_facts") or [],
            style_notes=structured.get("style_notes") or [],
            generated_at=datetime.now(timezone.utc).isoformat(),
            source_hash=source_digest,
            prompt_version=CHAPTER_MEMORY_PROMPT_VERSION,
            glossary_hash=glossary_digest,
            document_type=normalized_document_type,
        )
        cache_path.write_text(json.dumps(memory.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return memory


class RollingContextService:
    VALIDATED_STATUSES = ("TRANSLATED", "QA_PASSED")

    @classmethod
    def get_previous(
        cls,
        db: Session,
        chapter_id: str,
        before_order_index: int,
        max_nodes: int = 4,
        max_tokens: int = ROLLING_CONTEXT_MAX_TOKENS,
    ) -> List[BilingualContextItem]:
        rows = db.query(NodeModel).filter(
            NodeModel.chapter_id == chapter_id,
            NodeModel.order_index < before_order_index,
            NodeModel.status.in_(cls.VALIDATED_STATUSES),
            NodeModel.translated_content.isnot(None),
        ).order_by(NodeModel.order_index.desc()).limit(max_nodes * 2).all()
        return cls._bounded(list(reversed(rows)), max_nodes, max_tokens)

    @classmethod
    def get_neighbors(
        cls,
        db: Session,
        node: NodeModel,
        previous: int = 4,
        following: int = 2,
        max_tokens: int = ROLLING_CONTEXT_MAX_TOKENS,
    ) -> List[BilingualContextItem]:
        before = db.query(NodeModel).filter(
            NodeModel.chapter_id == node.chapter_id,
            NodeModel.order_index < node.order_index,
            NodeModel.status.in_(cls.VALIDATED_STATUSES),
            NodeModel.translated_content.isnot(None),
        ).order_by(NodeModel.order_index.desc()).limit(previous).all()
        after = db.query(NodeModel).filter(
            NodeModel.chapter_id == node.chapter_id,
            NodeModel.order_index > node.order_index,
            NodeModel.status.in_(cls.VALIDATED_STATUSES),
            NodeModel.translated_content.isnot(None),
        ).order_by(NodeModel.order_index.asc()).limit(following).all()
        return cls._bounded(list(reversed(before)), previous, max_tokens, following=after)

    @staticmethod
    def _bounded(
        rows: List[NodeModel],
        max_nodes: int,
        max_tokens: int,
        following: Optional[List[NodeModel]] = None,
    ) -> List[BilingualContextItem]:
        """Giữ đoạn liền trước trước tiên rồi mới lùi xa dần; đoạn liền sau chỉ lấy khi còn ngân sách.

        Trước đây duyệt từ đoạn cũ nhất nên hai đoạn dài đầu cửa sổ có thể đẩy mất đúng đoạn liền trước,
        là đoạn quan trọng nhất cho đại từ và mạch văn.
        """
        kept: List[NodeModel] = []
        used = 0

        def take(row: NodeModel) -> None:
            nonlocal used
            source, translation = _ending_of(row.content), _ending_of(row.translated_content or "")
            cost = max(1, int((len(source.split()) + len(translation.split())) * 1.45))
            if kept and used + cost > max_tokens:
                return
            kept.append(row)
            used += cost

        for row in reversed(rows[-max_nodes:]):
            take(row)
        for row in following or []:
            take(row)
        ordered = sorted(kept, key=lambda row: row.order_index)
        return [
            BilingualContextItem(row.id, _ending_of(row.content), _ending_of(row.translated_content or ""))
            for row in ordered
        ]
