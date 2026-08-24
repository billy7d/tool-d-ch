import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DOCUMENT_PROFILE_VERSION = "document-profile-v1"


@dataclass
class DocumentTranslationProfile:
    document_type: str = "GENERAL"
    domain: str = "general"
    intended_audience: str = "general readers"
    tone: str = "clear, neutral"
    register: str = "accessible"
    complexity: str = "medium"
    translation_preferences: Dict[str, Any] = field(default_factory=dict)
    proper_noun_policy: Dict[str, Any] = field(default_factory=dict)
    terminology_notes: List[str] = field(default_factory=list)
    style_notes: List[str] = field(default_factory=list)
    summary: str = ""
    version: str = DOCUMENT_PROFILE_VERSION
    generated_at: str = ""
    source_hash: str = ""
    prompt_version: str = DOCUMENT_PROFILE_VERSION
    setup_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "DocumentTranslationProfile":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value[key] for key in allowed if key in value})


def _node_text(node: Any) -> str:
    return str(getattr(node, "content", "") or "").strip()


class DocumentProfiler:
    @staticmethod
    def source_hash(nodes: Iterable[Any]) -> str:
        payload = "\n".join(_node_text(node) for node in nodes)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def stratified_sample(nodes: List[Any], max_chars: int = 7000) -> str:
        usable = [node for node in nodes if _node_text(node)]
        if not usable:
            return ""
        indexes = {0, len(usable) - 1}
        for ratio in (0.25, 0.5, 0.75):
            indexes.add(min(len(usable) - 1, int((len(usable) - 1) * ratio)))
        for index, node in enumerate(usable):
            node_type = str(getattr(node, "node_type", getattr(node, "type", ""))).lower()
            if "heading" in node_type:
                indexes.add(index)
        selected = [_node_text(usable[index]) for index in sorted(indexes)]
        return "\n\n".join(selected)[:max_chars]

    @staticmethod
    def fallback_profile(document_type: str, sample: str, custom_instructions: str = "") -> DocumentTranslationProfile:
        domain = (document_type or "GENERAL").upper()
        defaults = {
            "SELF_HELP": ("general adult readers", "direct, encouraging, conversational", "accessible"),
            "TECHNICAL": ("technical readers", "precise, instructional", "technical"),
            "ACADEMIC": ("academic readers", "formal, neutral", "scholarly"),
            "LEGAL": ("legal readers", "formal, conservative", "legal"),
            "LITERATURE": ("general readers", "narrative, voice-driven", "literary"),
            "BUSINESS": ("business readers", "concise, professional", "professional"),
            "FINANCE": ("finance readers", "precise, analytical", "professional"),
        }
        audience, tone, register = defaults.get(domain, ("general readers", "clear, neutral", "accessible"))
        sentence_count = max(1, len(re.findall(r"[.!?]+", sample)))
        complexity = "high" if len(sample.split()) / sentence_count > 28 else "medium"
        notes = [custom_instructions.strip()] if custom_instructions and custom_instructions.strip() else []
        return DocumentTranslationProfile(
            document_type=domain,
            domain=domain.lower().replace("_", " "),
            intended_audience=audience,
            tone=tone,
            register=register,
            complexity=complexity,
            translation_preferences={
                "prefer_short_sentences": domain in {"GENERAL", "SELF_HELP", "BUSINESS"},
                "preserve_examples": True,
                "sentence_restructuring": "conservative" if domain in {"LEGAL", "TECHNICAL", "ACADEMIC"} else "moderate",
            },
            proper_noun_policy={"preserve_names_products_acronyms": True, "translate_only_when_established": True},
            style_notes=notes,
            summary="Tài liệu thuộc lĩnh vực " + domain.lower().replace("_", " ") + ".",
        )

    @classmethod
    def load_or_create(
        cls,
        nodes: List[Any],
        cache_dir: Path,
        document_type: str,
        custom_instructions: str = "",
        provider: Optional[Any] = None,
        model_name: Optional[str] = None,
    ) -> DocumentTranslationProfile:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "document_profile.json"
        digest = cls.source_hash(nodes)
        setup_hash = hashlib.sha256(
            f"{(document_type or 'GENERAL').upper()}\n{custom_instructions.strip()}\n{DOCUMENT_PROFILE_VERSION}".encode("utf-8")
        ).hexdigest()
        if cache_path.exists():
            try:
                cached = DocumentTranslationProfile.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
                if cached.source_hash == digest and cached.version == DOCUMENT_PROFILE_VERSION and cached.setup_hash == setup_hash:
                    return cached
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        sample = cls.stratified_sample(nodes)
        profile = cls.fallback_profile(document_type, sample, custom_instructions)
        if provider and sample:
            try:
                summary = provider.summarize_context(sample, model=model_name)
                if summary:
                    profile.summary = summary[:2400]
            except Exception:
                pass
        profile.generated_at = datetime.now(timezone.utc).isoformat()
        profile.source_hash = digest
        profile.setup_hash = setup_hash
        cache_path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return profile
