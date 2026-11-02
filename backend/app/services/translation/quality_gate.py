import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from app.services.translation.glossary_validator import GlossaryValidator
from app.services.translation.language_validator import validate_target_language
from app.services.translation.numeric_validator import NumericValidator
from app.services.translation.polarity_validator import PolarityValidator
from app.services.translation.reference_validator import ReferenceValidator


URL_PATTERN = re.compile(r"https?://[^\s)\]}]+|www\.[^\s)\]}]+", re.IGNORECASE)
QUOTE_PATTERN = re.compile(r"[\"“”]")
OPENING_QUOTE_LOST = re.compile(r"^[^\"“”]+[,.!?…]\s*[\"”]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
# Câu ngắn trùng nhau là bình thường ("Anh gật đầu."); chỉ coi là chép khi câu đủ dài và gần như trùng hẳn.
CONTEXT_LEAK_MIN_WORDS = 6
CONTEXT_LEAK_OVERLAP = 0.8


# Token phải giữ nguyên từng ký tự: đường dẫn ổ đĩa/Unix, lời gọi hàm có dấu chấm, code inline.
# Chạy thật thấy qwen2.5:7b làm rơi "C:" khỏi "C:\workspace\app" mà không gate nào bắt.
TECHNICAL_TOKEN_PATTERN = re.compile(
    r"`[^`\n]+`"
    r"|[A-Za-z]:\\[^\s,;\"'“”()]+"
    r"|(?<![\w.])/(?:[\w.-]+/)+[\w.-]*"
    r"|\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\(\)"
)


# Từ tiếng Anh viết thường không trùng âm tiết tiếng Việt nào: còn trong bản dịch là chưa dịch hết.
# Chạy thật thấy qwen2.5:7b để lại "nobody chạm vào piano". Tránh các từ trùng âm tiết Việt
# ("to", "do", "in", "me", "an"...), từ kỹ thuật hay giữ nguyên ("build", "client") và chữ viết hoa.
ENGLISH_RESIDUE_WORDS = {
    "the", "and", "of", "with", "that", "this", "these", "those", "is", "are", "was", "were", "been",
    "have", "has", "had", "would", "could", "should", "nobody", "somebody", "everybody", "anybody",
    "nothing", "something", "everything", "anything", "never", "always", "said", "because", "which",
    "what", "when", "where", "there", "their", "they", "she", "his", "her", "him", "from", "into",
    # Không có "than": cũng là từ tiếng Việt (than thở, than phiền).
    "about", "then", "just", "very", "only", "also", "not", "however", "therefore", "although",
}
ENGLISH_WORD = re.compile(r"(?<![\w.`/\\-])[a-z]+(?![\w.`/\\-])")
EITHER_PATTERN = re.compile(r"\beither\b", re.IGNORECASE)
BOTH_PATTERN = re.compile(r"\bboth\b", re.IGNORECASE)
CA_HAI_PATTERN = re.compile(r"\bcả hai\b", re.IGNORECASE)
UNTRANSLATED_LABEL = re.compile(r"\b(?:Chapter|Section|Figure|Table|Appendix|Part)\s+\d+(?:\.\d+)*\b")


# Từ chức năng tiếng Anh viết hoa ở đầu câu mà từ kế tiếp viết thường: chạy trên sách thật thấy
# "Since sinh và tử dường như…". Từ kế tiếp viết hoa (tên sách "The Road Less Traveled") thì bỏ qua.
SENTENCE_START_ENGLISH = re.compile(
    r"(?:^|(?<=[.!?…:;]\s)|(?<=[“\"(]))"
    r"(Since|The|However|Although|Because|When|While|If|But|And|This|That|These|Those|There|It|What|Which|Nobody|Nothing|Everything)"
    r"\s+(?=[a-zà-ỹđ])"
)


_TOKEN = re.compile(r"[^\s“”\"()\[\]—–,;:.!?]+")


def _english_residue(text: str) -> List[str]:
    prose = URL_PATTERN.sub(" ", TECHNICAL_TOKEN_PATTERN.sub(" ", text or ""))
    tokens = [token.strip("'‘’") for token in _TOKEN.findall(prose)]
    tokens = [token for token in tokens if token]
    words = set()

    def neighbour(index: int, step: int) -> str:
        # Bỏ qua các từ chức năng liền nhau ("of the") để tới từ nội dung gần nhất.
        index += step
        while 0 <= index < len(tokens) and tokens[index] in ENGLISH_RESIDUE_WORDS:
            index += step
        return tokens[index] if 0 <= index < len(tokens) else ""

    for index, token in enumerate(tokens):
        if token not in ENGLISH_RESIDUE_WORDS:
            continue
        if neighbour(index, -1)[:1].isupper() and neighbour(index, 1)[:1].isupper():
            # Từ chức năng giữa hai từ viết hoa là một phần tên riêng/tựa sách giữ nguyên tiếng Anh
            # ("Psychology and Religion", "Journey of the Magi").
            continue
        words.add(token)
    words.update(match.group(1) for match in SENTENCE_START_ENGLISH.finditer(prose))
    return sorted(words)


def _technical_tokens(text: str) -> List[str]:
    return [token.rstrip(".,;:") for token in TECHNICAL_TOKEN_PATTERN.findall(text or "")]


def _word_set(text: str) -> Set[str]:
    return {word.casefold() for word in WORD_PATTERN.findall(text or "")}


def _expand_counter(values: Counter) -> List[str]:
    """Trả về các giá trị theo số lần xuất hiện để thông báo không làm mất bản sao."""
    expanded: List[str] = []
    for value, count in values.items():
        expanded.extend([value] * count)
    return expanded


def _url_counts(text: str) -> Counter:
    """Chuẩn hóa dấu câu kết câu để URL hợp lệ không bị coi là URL khác."""
    return Counter(url.rstrip(".,;:!?") for url in URL_PATTERN.findall(text or ""))


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    hard_fail: bool
    score: float
    issues: List[Dict[str, str]] = field(default_factory=list)


class TranslationQualityGate:
    """Tập trung các kiểm tra bắt buộc trước khi lưu bản dịch và TM."""

    @staticmethod
    def _issue(code: str, severity: str, message: str) -> Dict[str, str]:
        return {"code": code, "severity": severity, "message": message}

    @staticmethod
    def _content_addition_issues(source: str, target: str, context_translations: Sequence[str]) -> List[Dict[str, str]]:
        """Bắt các dạng chèn nội dung đã gặp khi chạy qwen2.5:7b thật.

        Model nhỏ hay kéo câu từ đoạn trước vào bản dịch (thêm câu hỏi, chép lại câu đã dịch)
        và làm rơi dấu ngoặc kép mở đầu lời thoại.
        """
        issues: List[Dict[str, str]] = []
        source_questions = URL_PATTERN.sub(" ", source).count("?")
        target_questions = URL_PATTERN.sub(" ", target).count("?")
        if target_questions > source_questions:
            issues.append(TranslationQualityGate._issue(
                "CONTENT_ADDITION", "ERROR",
                f"Bản dịch có {target_questions} câu hỏi trong khi nguồn có {source_questions}; có thể đã chèn nội dung.",
            ))
        source_quotes = len(QUOTE_PATTERN.findall(source))
        target_quotes = len(QUOTE_PATTERN.findall(target))
        # Nguồn mở đầu bằng lời thoại mà bản dịch lại bắt đầu bằng chữ rồi mới gặp dấu đóng ngoặc:
        # model đã rơi dấu mở (số dấu vẫn chẵn nếu rơi cả cặp nên kiểm đếm không bắt được).
        opening_lost = (
            source[:1] in "\"“" and target[:1] not in "\"“-–—" and OPENING_QUOTE_LOST.match(target)
        )
        if opening_lost or (target_quotes % 2 == 1 and source_quotes % 2 == 0):
            issues.append(TranslationQualityGate._issue(
                "QUOTE_UNBALANCED", "ERROR", "Dấu ngoặc kép trong bản dịch bị lệch (thiếu mở hoặc đóng lời thoại).",
            ))
        previous_sentences = [
            _word_set(sentence) for text in context_translations for sentence in SENTENCE_SPLIT.split(text or "")
        ]
        for sentence in SENTENCE_SPLIT.split(target):
            words = _word_set(sentence)
            if len(words) < CONTEXT_LEAK_MIN_WORDS:
                continue
            if any(len(words & other) / len(words | other) >= CONTEXT_LEAK_OVERLAP for other in previous_sentences if other):
                issues.append(TranslationQualityGate._issue(
                    "CONTEXT_LEAK", "ERROR", f"Câu \"{sentence.strip()[:80]}\" lặp lại bản dịch của đoạn trước.",
                ))
                break
        return issues

    def validate(
        self,
        source_text: str,
        translated_text: str,
        locked_glossary: Dict[str, str],
        context_translations: Optional[Sequence[str]] = None,
    ) -> QualityGateResult:
        source = (source_text or "").strip()
        target = (translated_text or "").strip()
        issues: List[Dict[str, str]] = []

        if not target:
            issues.append(self._issue("EMPTY_TRANSLATION", "ERROR", "Bản dịch đang để trống."))
        else:
            language = validate_target_language(source, target)
            if not language.passed:
                issues.append(self._issue(language.reason or "WRONG_TARGET_LANGUAGE", "ERROR", "Bản dịch không đạt yêu cầu ngôn ngữ đích."))

            numeric = NumericValidator.validate(source, target)
            if numeric.missing:
                missing_numbers = ", ".join(token.raw for token in numeric.missing)
                issues.append(self._issue("NUMBER_MISMATCH", "ERROR", f"Thiếu hoặc sai số liệu từ nguồn: {missing_numbers}"))
            if numeric.unexpected:
                unexpected_numbers = ", ".join(token.raw for token in numeric.unexpected)
                issues.append(self._issue("NUMBER_ADDITION", "ERROR", f"Bản dịch bổ sung số liệu không có trong nguồn: {unexpected_numbers}"))

            source_url_counts = _url_counts(source)
            target_url_counts = _url_counts(target)
            missing_urls = _expand_counter(source_url_counts - target_url_counts)
            unexpected_urls = _expand_counter(target_url_counts - source_url_counts)
            if missing_urls:
                issues.append(self._issue("URL_MISMATCH", "ERROR", f"Thiếu URL từ nguồn: {', '.join(missing_urls)}"))
            if unexpected_urls:
                issues.append(self._issue("URL_ADDITION", "ERROR", f"Bản dịch bổ sung URL không có trong nguồn: {', '.join(unexpected_urls)}"))

            references = ReferenceValidator.validate(source, target)
            if references.missing:
                missing_refs = ", ".join(token.raw for token in references.missing)
                issues.append(self._issue("REFERENCE_MISMATCH", "ERROR", f"Thiếu hoặc sai tham chiếu: {missing_refs}"))
            if references.unexpected:
                unexpected_refs = ", ".join(token.raw for token in references.unexpected)
                issues.append(self._issue("REFERENCE_ADDITION", "ERROR", f"Bản dịch bổ sung tham chiếu không có trong nguồn: {unexpected_refs}"))

            if len(source) > 100:
                ratio = len(target) / max(len(source), 1)
                if ratio < 0.25 or ratio > 3.5:
                    issues.append(self._issue("LENGTH_ANOMALY", "WARNING", f"Tỷ lệ độ dài bất thường: {ratio:.2f}."))

            glossary = GlossaryValidator.validate(source, target, locked_glossary or {})
            for violation in glossary.violations:
                issues.append(self._issue(
                    "GLOSSARY_MISMATCH",
                    "ERROR",
                    f"Thuật ngữ khóa '{violation.source_term}' phải dùng '{violation.expected_target}'.",
                ))
            for warning in glossary.warnings:
                allowed = ", ".join(warning.allowed_variants) or warning.expected_target
                issues.append(self._issue(
                    "GLOSSARY_PREFERENCE",
                    "WARNING",
                    f"Thuật ngữ mềm '{warning.source_term}' ưu tiên '{warning.expected_target}' (cho phép: {allowed}).",
                ))

            issues.extend(self._content_addition_issues(source, target, context_translations or []))

            if EITHER_PATTERN.search(source) and not BOTH_PATTERN.search(source) and CA_HAI_PATTERN.search(target):
                issues.append(self._issue(
                    "QUANTIFIER_SHIFT", "ERROR",
                    "Nguồn nói 'either' (bất kỳ/một trong hai) nhưng bản dịch thành 'cả hai'.",
                ))

            residue = _english_residue(target)
            if residue:
                issues.append(self._issue(
                    "ENGLISH_RESIDUE", "ERROR",
                    f"Còn từ tiếng Anh chưa dịch trong văn xuôi: {', '.join(residue[:5])}",
                ))
            untranslated_labels = [
                match.group(0) for match in UNTRANSLATED_LABEL.finditer(target)
                if not UNTRANSLATED_LABEL.search(source) or match.group(0) in source
            ]
            if untranslated_labels:
                issues.append(self._issue(
                    "UNTRANSLATED_LABEL", "ERROR",
                    f"Nhãn tham chiếu chưa dịch: {', '.join(untranslated_labels[:3])} (vd 'Chapter 1' -> 'Chương 1').",
                ))

            missing_tokens = [token for token in _technical_tokens(source) if token not in target]
            if missing_tokens:
                issues.append(self._issue(
                    "TECHNICAL_TOKEN_MISMATCH", "ERROR",
                    f"Thiếu hoặc bị sửa token kỹ thuật phải giữ nguyên: {', '.join(missing_tokens[:5])}",
                ))

            polarity = PolarityValidator.validate(source, target)
            if polarity.explicit_negation_lost:
                issues.append(self._issue("NEGATION_LOSS", "ERROR", "Bản dịch đã làm mất phủ định tường minh của nguồn."))
            elif polarity.ambiguous_negation_risk:
                issues.append(self._issue("NEGATION_RISK", "WARNING", "Bản dịch có nguy cơ làm mất ý phủ định."))

        hard_fail = any(issue["severity"] == "ERROR" for issue in issues)
        passed = not hard_fail
        penalty = sum(0.25 if issue["severity"] == "ERROR" else 0.1 for issue in issues)
        return QualityGateResult(passed, hard_fail, max(0.0, round(1.0 - penalty, 2)), issues)
