import re
from dataclasses import dataclass, field
from typing import Dict, List

from app.services.translation.glossary_validator import GlossaryValidator
from app.services.translation.language_validator import validate_target_language
from app.services.translation.numeric_validator import NumericValidator
from app.services.translation.polarity_validator import PolarityValidator
from app.services.translation.reference_validator import ReferenceValidator


URL_PATTERN = re.compile(r"https?://[^\s)\]}]+|www\.[^\s)\]}]+", re.IGNORECASE)
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

            numeric = NumericValidator.validate(source, target)
            if not numeric.passed:
                missing_numbers = ", ".join(token.raw for token in numeric.missing)
                issues.append(self._issue("NUMBER_MISMATCH", "ERROR", f"Thiếu hoặc sai số liệu từ nguồn: {missing_numbers}"))

            missing_urls = sorted(set(URL_PATTERN.findall(source)) - set(URL_PATTERN.findall(target)))
            if missing_urls:
                issues.append(self._issue("URL_MISMATCH", "ERROR", f"Thiếu URL từ nguồn: {', '.join(missing_urls)}"))

            references = ReferenceValidator.validate(source, target)
            if not references.passed:
                missing_refs = ", ".join(token.raw for token in references.missing)
                issues.append(self._issue("REFERENCE_MISMATCH", "ERROR", f"Thiếu hoặc sai tham chiếu: {missing_refs}"))

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

            polarity = PolarityValidator.validate(source, target)
            if polarity.explicit_negation_lost:
                issues.append(self._issue("NEGATION_LOSS", "ERROR", "Bản dịch đã làm mất phủ định tường minh của nguồn."))
            elif polarity.ambiguous_negation_risk:
                issues.append(self._issue("NEGATION_RISK", "WARNING", "Bản dịch có nguy cơ làm mất ý phủ định."))

        hard_fail = any(issue["severity"] == "ERROR" for issue in issues)
        passed = not hard_fail
        penalty = sum(0.25 if issue["severity"] == "ERROR" else 0.1 for issue in issues)
        return QualityGateResult(passed, hard_fail, max(0.0, round(1.0 - penalty, 2)), issues)
