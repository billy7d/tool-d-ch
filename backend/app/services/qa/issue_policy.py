from dataclasses import dataclass
from typing import Dict, Iterable, Optional


DETERMINISTIC_ISSUES = frozenset({
    "EMPTY_TRANSLATION",
    "NUMBER_MISMATCH",
    "URL_MISMATCH",
    "BROKEN_REFERENCE",
    "REFERENCE_MISMATCH",
    "GLOSSARY_MISMATCH",
    "WRONG_TARGET_LANGUAGE",
    "FOREIGN_SCRIPT_CONTAMINATION",
    "CONTAINS_CHINESE",
    "NEGATION_LOSS",
})

SEMANTIC_ISSUES = frozenset({
    "AI_QA",
    "AI_QA_RECOMMENDATION",
    "AI_SEMANTIC_ERROR",
    "SEMANTIC_MISMATCH",
    "MEANING_ERROR",
    "OMISSION",
    "HALLUCINATION",
    "CAUSALITY_ERROR",
    "POLARITY_ERROR",
    "POSSIBLE_TRUNCATION",
    "POSSIBLE_ADDED_CONTENT",
})

ISSUE_ALIASES = {
    "BROKEN_REFERENCE": "URL_MISMATCH",
    "CONTAINS_CHINESE": "FOREIGN_SCRIPT_CONTAMINATION",
}


@dataclass(frozen=True)
class IssueResolution:
    statuses: Dict[str, str]
    semantic_review_required: bool
    qa_error: Optional[str] = None


def classify_issue(issue_type: str) -> str:
    normalized = (issue_type or "").strip().upper()
    if normalized in DETERMINISTIC_ISSUES:
        return "DETERMINISTIC"
    return "SEMANTIC"


def evaluate_issue_resolution(
    issue_types: Iterable[str],
    remaining_deterministic_issues: Iterable[str],
    semantic_review: Optional[dict] = None,
) -> IssueResolution:
    normalized_remaining = {
        ISSUE_ALIASES.get((code or "").strip().upper(), (code or "").strip().upper())
        for code in remaining_deterministic_issues
    }
    normalized_types = [(issue_type or "").strip().upper() for issue_type in issue_types]
    semantic_required = any(classify_issue(issue_type) == "SEMANTIC" for issue_type in normalized_types)
    qa_error: Optional[str] = None
    semantic_passed = False
    if semantic_required:
        if not isinstance(semantic_review, dict):
            qa_error = "Thiếu kết quả semantic re-review hợp lệ."
        elif semantic_review.get("status") == "ERROR" or semantic_review.get("error"):
            qa_error = str(semantic_review.get("error") or "Semantic re-review thất bại.")
        else:
            semantic_passed = semantic_review.get("status") == "PASS" and semantic_review.get("is_passed") is True

    statuses: Dict[str, str] = {}
    for issue_type in normalized_types:
        if classify_issue(issue_type) == "DETERMINISTIC":
            canonical = ISSUE_ALIASES.get(issue_type, issue_type)
            statuses[issue_type] = "OPEN" if canonical in normalized_remaining else "RESOLVED"
        else:
            statuses[issue_type] = "RESOLVED" if semantic_passed else "OPEN"
    return IssueResolution(statuses, semantic_required, qa_error)
