from dataclasses import asdict, dataclass, field
from typing import Dict, List

from app.services.translation.term_matcher import TermMatcher


@dataclass(frozen=True)
class GlossaryViolation:
    source_term: str
    expected_target: str
    reason: str = "TARGET_TERM_MISSING"


@dataclass(frozen=True)
class GlossaryValidationResult:
    passed: bool
    violations: List[GlossaryViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "violations": [asdict(violation) for violation in self.violations],
        }


class GlossaryValidator:
    """Kiểm tra thuật ngữ khóa mà không tự ý sửa nội dung bản dịch."""

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        return TermMatcher.contains(text, term)

    @classmethod
    def validate(
        cls,
        source_text: str,
        translated_text: str,
        glossary_terms: Dict[str, str],
    ) -> GlossaryValidationResult:
        violations: List[GlossaryViolation] = []
        for source_term, expected_target in (glossary_terms or {}).items():
            if not cls._contains_term(source_text, source_term):
                continue
            if not cls._contains_term(translated_text, expected_target):
                violations.append(
                    GlossaryViolation(
                        source_term=source_term,
                        expected_target=expected_target,
                    )
                )
        return GlossaryValidationResult(passed=not violations, violations=violations)
