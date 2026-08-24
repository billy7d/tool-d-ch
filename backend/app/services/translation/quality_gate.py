import re
from dataclasses import dataclass, field
from typing import Dict, List

from app.services.translation.glossary_validator import GlossaryValidator
from app.services.translation.language_validator import validate_target_language


NUMBER_PATTERN = re.compile(
    r"(?:[$€£]\s*)?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?%?"
    r"|(?:[$€£]\s*)?\d+(?:[.,]\d+)?%?"
)
URL_PATTERN = re.compile(r"https?://[^\s)\]}]+|www\.[^\s)\]}]+", re.IGNORECASE)
REFERENCE_PATTERN = re.compile(
    r"\b(?:Figure|Fig\.?|Table|Chapter|Section)\s+\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)


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
    def _number_key(value: str) -> str:
        suffix = "%" if "%" in value else ""
        currency = "".join(char for char in value if char in "$€£")
        digits = re.sub(r"\D", "", value)
        return f"{currency}{digits}{suffix}"

    def validate(
        self,
        source_text: str,
        translated_text: str,
        locked_glossary: Dict[str, str],
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

            source_numbers = {self._number_key(value): value for value in NUMBER_PATTERN.findall(source)}
            target_numbers = {self._number_key(value) for value in NUMBER_PATTERN.findall(target)}
            missing_numbers = [raw for key, raw in source_numbers.items() if key not in target_numbers]
            if missing_numbers:
                issues.append(self._issue("NUMBER_MISMATCH", "ERROR", f"Thiếu số liệu từ nguồn: {', '.join(missing_numbers)}"))

            missing_urls = sorted(set(URL_PATTERN.findall(source)) - set(URL_PATTERN.findall(target)))
            if missing_urls:
                issues.append(self._issue("URL_MISMATCH", "ERROR", f"Thiếu URL từ nguồn: {', '.join(missing_urls)}"))

            source_refs = set(REFERENCE_PATTERN.findall(source))
            target_ref_numbers = {re.search(r"\d+(?:\.\d+)*", ref).group(0) for ref in REFERENCE_PATTERN.findall(target)}
            missing_refs = [ref for ref in source_refs if re.search(r"\d+(?:\.\d+)*", ref).group(0) not in target_ref_numbers]
            if missing_refs:
                issues.append(self._issue("REFERENCE_MISMATCH", "ERROR", f"Thiếu tham chiếu: {', '.join(missing_refs)}"))

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

        hard_fail = any(issue["severity"] == "ERROR" for issue in issues)
        passed = not issues
        penalty = sum(0.25 if issue["severity"] == "ERROR" else 0.1 for issue in issues)
        return QualityGateResult(passed, hard_fail, max(0.0, round(1.0 - penalty, 2)), issues)
