"""Bộ chọn few-shot deterministic theo ngữ cảnh hiện tại."""

import re
from typing import Dict, Iterable, List, Sequence, Set

from app.services.translation.few_shot_library import FewShotExample, curated_examples


PATTERN_NAMES = {
    "passive", "nominalization", "long_noun_phrase", "relative_clause",
    "conditional", "modality", "causality", "pronoun_chain", "idiom",
    "phrasal_verb", "business_collocation", "technical_definition",
    "legal_obligation", "academic_hedging", "sentence_split", "sentence_merge",
    "discourse_connector",
}


class FewShotSelector:
    """Chọn tối đa bốn ví dụ, không gọi model và luôn có thứ tự ổn định."""

    DEFAULT_LIMIT = 3
    MAX_LIMIT = 4

    @staticmethod
    def detect_patterns(source_text: str) -> Set[str]:
        text = (source_text or "").strip()
        lowered = text.casefold()
        patterns: Set[str] = set()
        if re.search(r"\b(?:is|are|was|were|be|been|being|get|got)\s+\w+(?:ed|en)\b", lowered):
            patterns.add("passive")
        if re.search(r"\b\w+(?:tion|sion|ment|ity|ance|ence|al)\b", lowered):
            patterns.add("nominalization")
        if re.search(r"\bthe\s+[^.!?;]{8,}\s+of\s+[^.!?;]{3,}", lowered):
            patterns.add("long_noun_phrase")
        if re.search(r"\b(?:which|who|whom|whose|where|that)\b", lowered):
            patterns.add("relative_clause")
        if re.search(r"\b(?:if|unless|provided that|as long as|whether)\b", lowered):
            patterns.add("conditional")
        if re.search(r"\b(?:may|might|must|should|can|could|shall|would|will|need to)\b", lowered):
            patterns.add("modality")
        if re.search(r"\b(?:because|therefore|thus|hence|so that|as a result|due to|resulted from)\b", lowered):
            patterns.add("causality")
        if re.search(r"\b(?:he|she|they|it|them|her|him|their|his|its|this|that)\b", lowered):
            patterns.add("pronoun_chain")
        if re.search(r"\b(?:break the ice|on the same page|give in|by heart|take with a grain of salt)\b", lowered):
            patterns.add("idiom")
        if re.search(r"\b(?:look into|set up|follow up|take over|give rise to|carry out|point out|fall back|write down)\b", lowered):
            patterns.add("phrasal_verb")
        if re.search(r"\b(?:revenue|margin|forecast|stakeholder|churn|sales|budget|client|board|strategy|operational)\b", lowered):
            patterns.add("business_collocation")
        if re.search(r"\b(?:is|are|means|refers to|defined as|component that|function returns|schema)\b", lowered):
            patterns.add("technical_definition")
        if re.search(r"\b(?:shall|must not|required|prohibited|agreement|licensee|supplier|warranty|statutory)\b", lowered):
            patterns.add("legal_obligation")
        if re.search(r"\b(?:may suggest|suggests|likely|appears|could|not necessarily|broadly consistent)\b", lowered):
            patterns.add("academic_hedging")
        if len(re.findall(r"[.!?]", text)) > 1 or re.search(r"\b(?:then|after that|first),?\b", lowered):
            patterns.add("sentence_split")
        if re.search(r"[,;]\s*(?:and|but|yet|while|so)\b", lowered) or re.search(r"\band\b", lowered):
            patterns.add("sentence_merge")
        if re.search(r"\b(?:however|although|although|nevertheless|therefore|first|then|finally|except)\b", lowered):
            patterns.add("discourse_connector")
        return patterns & PATTERN_NAMES

    @staticmethod
    def _node_type(value: object) -> str:
        return str(getattr(value, "value", value or "paragraph")).casefold()

    @classmethod
    def _score(
        cls,
        example: FewShotExample,
        domain: str,
        mode: str,
        node_type: str,
        source_text: str,
        source_patterns: Set[str],
    ) -> int:
        score = 100 if example.domain == domain else 0
        if example.node_type.casefold() == node_type:
            score += 30
        elif node_type in {"paragraph", "quote"} and example.node_type == "paragraph":
            score += 8
        if mode in example.modes:
            score += 12
        score += 18 * len(source_patterns.intersection(example.patterns))
        source_length = len((source_text or "").split())
        example_length = len(example.source.split())
        score += max(0, 12 - min(12, abs(source_length - example_length) // 4))
        if mode in {"FAITHFUL", "ACADEMIC", "TECHNICAL"} and example.register in {"formal", "scholarly", "precise"}:
            score += 5
        return score

    @classmethod
    def select(
        cls,
        domain: str,
        mode: str = "NATURAL",
        node_type: object = "paragraph",
        source_text: str = "",
        source_patterns: Iterable[str] | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> List[Dict[str, object]]:
        normalized_domain = (domain or "GENERAL").upper()
        if normalized_domain not in {item.domain for item in curated_examples(normalized_domain)}:
            normalized_domain = "GENERAL"
        normalized_mode = str(getattr(mode, "value", mode or "NATURAL")).upper()
        normalized_node_type = cls._node_type(node_type)
        detected = set(source_patterns or ()) | cls.detect_patterns(source_text)
        maximum = min(cls.MAX_LIMIT, max(0, int(limit)))
        if maximum == 0:
            return []
        candidates = curated_examples(normalized_domain)
        ranked = sorted(
            candidates,
            key=lambda item: (
                -cls._score(item, normalized_domain, normalized_mode, normalized_node_type, source_text, detected),
                item.example_id,
            ),
        )
        selected: List[Dict[str, object]] = []
        seen_sources = set()
        for example in ranked:
            if example.source in seen_sources:
                continue
            seen_sources.add(example.source)
            selected.append(example.to_dict())
            if len(selected) >= maximum:
                break
        return selected


def select_few_shots(
    domain: str,
    mode: str = "NATURAL",
    node_type: object = "paragraph",
    limit: int = FewShotSelector.DEFAULT_LIMIT,
    source_text: str = "",
    source_patterns: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    """API module-level giữ tương thích với selector P0/P1 cũ."""
    return FewShotSelector.select(
        domain=domain,
        mode=mode,
        node_type=node_type,
        source_text=source_text,
        source_patterns=source_patterns,
        limit=limit,
    )
