import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Set

from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor


VIETNAMESE_STOPWORDS = {
    "và", "là", "của", "có", "được", "cho", "trong", "với", "không",
    "một", "những", "các", "đã", "đang", "này", "đó", "từ", "đến",
    "khi", "về", "theo", "nhưng", "cũng", "sẽ", "tại", "trên", "dưới",
    "công", "ty", "doanh", "thu", "phát", "hành", "phiên", "bản", "mới",
    "sử", "dụng", "để", "tăng", "giảm", "năm", "người", "việc", "thành",
    "nay", "hoat", "dong", "khong", "duoc", "can", "phai", "nen", "da",
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "to", "of", "in", "on", "for", "with", "from", "by",
    "this", "that", "these", "those", "it", "its", "as", "at", "not",
    "also", "company", "increased", "revenue", "expanded", "overseas",
    "can", "cannot", "could", "would", "should", "must", "shall", "may",
    "have", "has", "had", "do", "does", "did", "will", "into", "through",
    "enables", "between", "without", "within", "after", "before", "during",
}

WORD_PATTERN = re.compile(r"[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*", re.UNICODE)
VIETNAMESE_DIACRITIC_PATTERN = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    r"íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
URL_OR_IDENTIFIER_PATTERN = re.compile(
    r"(?:https?://\S+|www\.\S+|\b[A-Z][A-Z0-9_+.#/-]{1,}\b|\b\w+[_/.]\w+\b)"
)


@dataclass(frozen=True)
class TargetLanguageValidationResult:
    passed: bool
    reason: Optional[str]
    vietnamese_score: float
    english_residual_score: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _unexpected_cjk(source_text: str, translated_text: str) -> Set[str]:
    source_chars = set(VietnamesePostProcessor.CHINESE_PATTERN.findall(source_text or ""))
    target_chars = set(VietnamesePostProcessor.CHINESE_PATTERN.findall(translated_text or ""))
    return target_chars - source_chars


def validate_target_language(
    source_text: str,
    translated_text: str,
    target_language: str = "vi",
) -> TargetLanguageValidationResult:
    """Phát hiện đầu ra sai ngôn ngữ bằng heuristic nhẹ và có chủ đích bảo thủ."""
    text = (translated_text or "").strip()
    if not text:
        return TargetLanguageValidationResult(False, "EMPTY_TRANSLATION", 0.0, 0.0)
    if target_language.lower() != "vi":
        return TargetLanguageValidationResult(True, None, 1.0, 0.0)
    if _unexpected_cjk(source_text, text):
        return TargetLanguageValidationResult(False, "FOREIGN_SCRIPT_CONTAMINATION", 0.0, 0.0)

    filtered = URL_OR_IDENTIFIER_PATTERN.sub(" ", text)
    words = [word.lower() for word in WORD_PATTERN.findall(filtered)]
    if not words:
        return TargetLanguageValidationResult(True, None, 1.0, 0.0)

    vi_hits = sum(word in VIETNAMESE_STOPWORDS for word in words)
    en_hits = sum(word in ENGLISH_STOPWORDS for word in words)
    diacritic_hits = sum(bool(VIETNAMESE_DIACRITIC_PATTERN.search(word)) for word in words)
    total = len(words)
    vietnamese_score = min(1.0, (vi_hits * 1.4 + diacritic_hits * 1.1) / max(total, 1))
    english_score = min(1.0, en_hits / max(total, 1))

    clearly_english = en_hits >= 2 and en_hits > (vi_hits + diacritic_hits) * 1.35
    mixed_english = total >= 8 and en_hits >= 3 and english_score >= 0.28 and vietnamese_score < 0.35
    source_words = {word.lower() for word in WORD_PATTERN.findall(URL_OR_IDENTIFIER_PATTERN.sub(" ", source_text or ""))}
    lexical_overlap = sum(word in source_words for word in words) / max(total, 1)
    untranslated_phrase = total >= 5 and lexical_overlap >= 0.8 and vi_hits == 0 and diacritic_hits == 0
    if clearly_english or mixed_english or untranslated_phrase:
        return TargetLanguageValidationResult(
            False,
            "WRONG_TARGET_LANGUAGE",
            round(vietnamese_score, 4),
            round(english_score, 4),
        )

    return TargetLanguageValidationResult(
        True,
        None,
        round(vietnamese_score, 4),
        round(english_score, 4),
    )
