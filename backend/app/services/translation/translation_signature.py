import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict


PROMPT_VERSION = "translation-v3-contextual"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TranslationSignature:
    style_hash: str
    glossary_hash: str
    prompt_version: str = PROMPT_VERSION


def build_translation_signature(
    source_language: str,
    target_language: str,
    translation_mode: str,
    document_type: str,
    style_guide: Dict[str, Any],
    locked_glossary: Dict[str, str],
    custom_instructions: str = "",
    style_pack_version: str = "style-packs-v1",
    prompt_version: str = PROMPT_VERSION,
) -> TranslationSignature:
    style_hash = canonical_hash({
        "source_language": source_language,
        "target_language": target_language,
        "translation_mode": translation_mode,
        "document_type": document_type,
        "style_guide": style_guide or {},
        "custom_instructions": custom_instructions or "",
        "style_pack_version": style_pack_version,
        "prompt_version": prompt_version,
    })
    glossary_hash = canonical_hash(locked_glossary or {})
    return TranslationSignature(style_hash, glossary_hash, prompt_version)
