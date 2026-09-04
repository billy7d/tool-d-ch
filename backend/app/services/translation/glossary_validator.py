from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List

from app.services.translation.term_matcher import TermMatcher


@dataclass(frozen=True)
class GlossaryViolation:
    source_term: str
    expected_target: str
    reason: str = "TARGET_TERM_MISSING"
    lock_level: str = "HARD"
    allowed_variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class GlossaryValidationResult:
    passed: bool
    violations: List[GlossaryViolation] = field(default_factory=list)
    warnings: List[GlossaryViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "violations": [asdict(violation) for violation in self.violations],
            "warnings": [asdict(warning) for warning in self.warnings],
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
        glossary_terms: Any,
    ) -> GlossaryValidationResult:
        violations: List[GlossaryViolation] = []
        warnings: List[GlossaryViolation] = []
        for source_term, expected_target, lock_level, allowed_variants in cls._entries(glossary_terms):
            if not cls._contains_term(source_text, source_term):
                continue
            if cls._contains_term(translated_text, expected_target) or any(
                cls._contains_term(translated_text, variant) for variant in allowed_variants
            ):
                continue
            issue = GlossaryViolation(
                source_term=source_term,
                expected_target=expected_target,
                lock_level=lock_level,
                allowed_variants=allowed_variants,
            )
            if lock_level == "HARD":
                violations.append(issue)
            else:
                warnings.append(issue)
        return GlossaryValidationResult(passed=not violations, violations=violations, warnings=warnings)

    @classmethod
    def _entries(cls, glossary_terms: Any) -> Iterable[tuple[str, str, str, tuple[str, ...]]]:
        """Nhận cả map legacy lẫn record glossary có sense/lock_level."""
        if isinstance(glossary_terms, Mapping):
            for source_term, target in glossary_terms.items():
                if isinstance(target, Mapping):
                    yield cls._entry_from_mapping(str(source_term), target)
                else:
                    yield str(source_term), str(target or ""), "HARD", ()
            return
        for item in glossary_terms or []:
            if isinstance(item, Mapping):
                source_term = str(item.get("source_term", item.get("source", "")) or "").strip()
                yield cls._entry_from_mapping(source_term, item)

    @staticmethod
    def _entry_from_mapping(source_term: str, item: Mapping[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
        expected = str(
            item.get("preferred_target")
            or item.get("target_term")
            or item.get("target")
            or ""
        ).strip()
        raw_level = str(item.get("lock_level", "") or "").upper()
        if raw_level not in {"HARD", "SOFT"}:
            raw_level = "HARD" if bool(item.get("locked", True)) else "SOFT"
        variants = item.get("allowed_variants") or item.get("variants") or []
        if isinstance(variants, str):
            variants = [variants]
        normalized_variants = tuple(str(value).strip() for value in variants if str(value).strip())
        return source_term, expected, raw_level, normalized_variants
