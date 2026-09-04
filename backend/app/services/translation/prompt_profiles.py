from dataclasses import dataclass
from typing import Dict, List

from app.services.translation.few_shot_library import (
    FEW_SHOT_LIBRARY_VERSION,
    SUPPORTED_DOMAINS,
    curated_examples_as_dict,
)
from app.services.translation.few_shot_selector import select_few_shots


STYLE_PACK_VERSION = "style-packs-v1"


@dataclass(frozen=True)
class StylePack:
    domain: str
    register: str
    instructions: str
    forbidden: str


STYLE_PACKS: Dict[str, StylePack] = {
    "GENERAL": StylePack("GENERAL", "neutral, accessible", "Write clear, natural and accessible Vietnamese. Prefer common Vietnamese syntax and concise verbs.", "Avoid needless academic or Sino-Vietnamese wording and literal English syntax."),
    "BUSINESS": StylePack("BUSINESS", "professional", "Use concise professional Vietnamese and established business terminology while keeping the prose readable.", "Do not turn ordinary prose into stiff corporate jargon."),
    "FINANCE": StylePack("FINANCE", "precise", "Prioritize standard financial terminology and preserve every number, ratio, metric, unit and degree of certainty.", "Do not naturalize wording in a way that changes financial meaning."),
    "SELF_HELP": StylePack("SELF_HELP", "accessible, encouraging", "Use friendly, direct and motivating Vietnamese with readable sentence length.", "Avoid research-like prose, heavy vocabulary and exaggerated promises."),
    "TECHNICAL": StylePack("TECHNICAL", "precise, practical", "Preserve technical accuracy and standard terminology. Keep APIs, commands, identifiers, class and function names unchanged.", "Do not translate code or invent technical explanations."),
    "ACADEMIC": StylePack("ACADEMIC", "formal, scholarly", "Use formal, precise, neutral Vietnamese appropriate for scholarly publication.", "Do not make the sentence harder than the source or add claims."),
    "LEGAL": StylePack("LEGAL", "formal, conservative", "Preserve scope, modality, conditions, obligations, prohibitions and defined terms exactly. Restructure conservatively.", "Do not soften or strengthen legal force."),
    "LITERATURE": StylePack("LITERATURE", "voice-driven", "Preserve narrative voice, rhythm, dialogue, characterization and emotional nuance in idiomatic Vietnamese.", "Do not embellish facts or rewrite the plot."),
}


# Giữ tên export cũ cho benchmark P0/P1, nhưng dữ liệu nay đến từ thư viện curated.
FEW_SHOTS: Dict[str, List[dict]] = {
    domain: curated_examples_as_dict(domain) for domain in SUPPORTED_DOMAINS
}


def get_style_pack(domain: str) -> StylePack:
    return STYLE_PACKS.get((domain or "GENERAL").upper(), STYLE_PACKS["GENERAL"])


# `select_few_shots` được import từ few_shot_selector để các caller cũ vẫn dùng
# đúng hàm nhưng đã có scoring theo source pattern và giới hạn tối đa bốn ví dụ.
