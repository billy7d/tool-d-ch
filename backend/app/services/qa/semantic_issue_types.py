from typing import Dict


SEMANTIC_ISSUES = {
    "SEMANTIC_OMISSION",
    "SEMANTIC_ADDITION",
    "MEANING_DRIFT",
    "MODALITY_ERROR",
    "CAUSALITY_ERROR",
    "SCOPE_ERROR",
    "CONDITION_ERROR",
    "COMPARISON_ERROR",
    "ENTITY_REFERENCE_ERROR",
    "PRONOUN_AMBIGUITY",
}

CONSISTENCY_ISSUES = {
    "ENTITY_MISMATCH",
    "ENTITY_VARIANT",
    "TERM_VARIANT",
    "ACRONYM_INCONSISTENCY",
    "REPEATED_PHRASE_INCONSISTENCY",
}

SYSTEM_ISSUES = {"SEMANTIC_QA_ERROR", "CONTEXT_BUDGET_EXCEEDED", "PROVIDER_ERROR"}

_ALIASES: Dict[str, str] = {
    "OMISSION": "SEMANTIC_OMISSION",
    "MISSING_MEANING": "SEMANTIC_OMISSION",
    "ADDITION": "SEMANTIC_ADDITION",
    "HALLUCINATION": "SEMANTIC_ADDITION",
    "SEMANTIC_MISMATCH": "MEANING_DRIFT",
    "QA_ERROR": "SEMANTIC_QA_ERROR",
    "ENTITY_CONSISTENCY": "ENTITY_VARIANT",
    "LOCKED_ENTITY_MISMATCH": "ENTITY_MISMATCH",
}

_WARNING_TYPES = {"PRONOUN_AMBIGUITY", "ENTITY_VARIANT", "TERM_VARIANT", "ACRONYM_INCONSISTENCY", "REPEATED_PHRASE_INCONSISTENCY"}


def normalize_semantic_issue_type(value: str) -> str:
    """Chuẩn hóa tên lỗi provider về taxonomy Phase 3 duy nhất."""
    normalized = str(value or "MEANING_DRIFT").strip().upper().replace(" ", "_")
    return _ALIASES.get(normalized, normalized if normalized in SEMANTIC_ISSUES | CONSISTENCY_ISSUES | SYSTEM_ISSUES else "MEANING_DRIFT")


def semantic_severity(issue_type: str) -> str:
    return "WARNING" if normalize_semantic_issue_type(issue_type) in _WARNING_TYPES else "ERROR"
