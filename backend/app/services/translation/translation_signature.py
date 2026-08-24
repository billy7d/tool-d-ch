import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict
from app.services.translation.translation_config import TranslationConfig


PROMPT_VERSION = "translation-v3.1-stable"


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
    register: str = "",
    sentence_style: str = "",
) -> TranslationSignature:
    style_hash = canonical_hash({
        "source_language": source_language,
        "target_language": target_language,
        "translation_mode": translation_mode,
        "document_type": document_type,
        "style_guide": style_guide or {},
        "register": (register or "").upper(),
        "sentence_style": (sentence_style or "").upper(),
        "custom_instructions": custom_instructions or "",
        "style_pack_version": style_pack_version,
        "prompt_version": prompt_version,
    })
    glossary_hash = canonical_hash(locked_glossary or {})
    return TranslationSignature(style_hash, glossary_hash, prompt_version)


def build_translation_signature_from_config(
    config: TranslationConfig,
    locked_glossary: Dict[str, str],
    prompt_version: str = PROMPT_VERSION,
) -> TranslationSignature:
    return build_translation_signature(
        source_language=config.source_language,
        target_language=config.target_language,
        translation_mode=config.translation_mode,
        document_type=config.document_type,
        style_guide=config.style_guide,
        locked_glossary=locked_glossary,
        custom_instructions=config.custom_instructions,
        prompt_version=prompt_version,
        register=config.register,
        sentence_style=config.sentence_style,
    )
