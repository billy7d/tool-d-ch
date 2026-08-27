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
CHAPTER_MEMORY_PROMPT_VERSION = "chapter-memory-prompt-v3-json"
CHAPTER_MEMORY_FIELDS = {
    "summary", "entities", "key_concepts", "tone", "pronoun_notes",
    "terminology", "important_facts", "style_notes",
}
CHAPTER_MEMORY_LIST_LIMITS = {
    "entities": 20,
    "key_concepts": 15,
    "pronoun_notes": 10,
    "terminology": 30,
    "important_facts": 15,
    "style_notes": 10,
}


def validate_chapter_memory_response(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or set(value) != CHAPTER_MEMORY_FIELDS:
        raise ValueError("Chapter Memory phải là object đúng schema.")
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

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ChapterMemory":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value[key] for key in allowed if key in value})


@dataclass(frozen=True)
class BilingualContextItem:
    node_id: str
    source: str
    translation: str


class ChapterMemoryBuilder:
    @staticmethod
    def source_hash(nodes: List[Any]) -> str:
        text = "\n".join(str(getattr(node, "content", "") or "") for node in nodes)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def stratified_sample(nodes: List[Any], max_chars: int = 5500) -> str:
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
        for name in re.findall(r"\b[A-Z][A-Za-z.&'-]+(?:\s+[A-Z][A-Za-z.&'-]+){0,3}\b", sample):
            if name.lower() not in seen and len(name) > 2:
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
        summary = (sample[:600].strip() or f"Chương '{chapter_title}' chưa có nội dung tóm tắt.")
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
        max_tokens: int = 700,
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
        previous: int = 2,
        following: int = 2,
        max_tokens: int = 700,
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
        return cls._bounded(list(reversed(before)) + after, previous + following, max_tokens)

    @staticmethod
    def _bounded(rows: List[NodeModel], max_nodes: int, max_tokens: int) -> List[BilingualContextItem]:
        selected: List[BilingualContextItem] = []
        used = 0
        for row in rows[-max_nodes:]:
            cost = max(1, int((len(row.content.split()) + len((row.translated_content or "").split())) * 1.45))
            if selected and used + cost > max_tokens:
                continue
            selected.append(BilingualContextItem(row.id, row.content, row.translated_content or ""))
            used += cost
        return selected
