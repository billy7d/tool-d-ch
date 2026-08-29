import re
from dataclasses import dataclass
from typing import Iterable, Optional


SEMANTIC_POLICY_VERSION = "semantic-policy-v1"


@dataclass(frozen=True)
class SemanticRiskResult:
    score: float
    level: str
    reasons: list[str]
    requires_critic: bool


class SemanticRiskScorer:
    """Chấm risk thuần deterministic; tuyệt đối không gọi provider."""

    MEDIUM_THRESHOLD = 0.35
    HIGH_THRESHOLD = 0.65
    SENSITIVE = {
        "condition": r"\b(if|unless|only if|provided that|except when)\b",
        "modality": r"\b(may|might|must|shall|should|could|would|likely|unlikely)\b",
        "causality": r"\b(because|because of|therefore|thus|consequently|as a result)\b",
        "scope": r"\b(not only|at least|at most|only|all|none|each|every)\b",
        "contrast": r"\b(although|however|despite|whereas|rather than|instead of)\b",
        "comparison": r"\b(more than|less than|higher than|lower than|compared with|approximately)\b",
    }
    HIGH_RISK_TYPES = {"legal", "academic", "finance", "technical", "footnote", "table"}

    @classmethod
    def score(
        cls,
        source_text: str,
        translated_text: str = "",
        node_type: str = "paragraph",
        document_type: str = "GENERAL",
        previous_repairs: int = 0,
        qa_warnings: Optional[Iterable[str]] = None,
        entity_count: int = 0,
        medium_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
    ) -> SemanticRiskResult:
        source = source_text or ""
        target = translated_text or ""
        reasons: list[str] = []
        value = 0.05
        words = len(source.split())
        if words >= 35:
            value += 0.16
            reasons.append("LONG_SOURCE")
        if words >= 65:
            value += 0.12
            reasons.append("VERY_LONG_SOURCE")
        clauses = len(re.findall(r"[,;:]|\b(and|or|but|which|that|while)\b", source, re.I))
        if clauses >= 3:
            value += min(0.18, clauses * 0.03)
            reasons.append("MULTI_CLAUSE")
        for label, pattern in cls.SENSITIVE.items():
            if re.search(pattern, source, re.I):
                value += 0.18 if label in {"condition", "modality"} else 0.12
                reasons.append(label.upper())
        if str(document_type).lower() in cls.HIGH_RISK_TYPES or str(node_type).lower() in cls.HIGH_RISK_TYPES:
            value += 0.25
            reasons.append("SENSITIVE_DOMAIN_OR_NODE")
        if previous_repairs:
            value += min(0.30, 0.18 * previous_repairs)
            reasons.append("PREVIOUS_REPAIR")
        warnings = {str(item).upper() for item in (qa_warnings or [])}
        if warnings & {"NEGATION_RISK", "LENGTH_ANOMALY"}:
            reasons.append("QA_WARNING")
        # NEGATION_RISK là cảnh báo ngữ nghĩa nhạy cảm: phải đủ đẩy câu ngắn
        # lên MEDIUM để không bị bỏ qua chỉ vì critic wiring bị thiếu.
        if "NEGATION_RISK" in warnings:
            value += 0.30
            reasons.append("QA_WARNING_NEGATION_RISK")
        if "LENGTH_ANOMALY" in warnings:
            value += 0.16
            reasons.append("QA_WARNING_LENGTH_ANOMALY")
        if entity_count >= 3:
            value += 0.10
            reasons.append("ENTITY_DENSITY")
        if source and target:
            source_sentences = max(1, len(re.findall(r"[.!?]+", source)))
            target_sentences = max(1, len(re.findall(r"[.!?]+", target)))
            if abs(source_sentences - target_sentences) >= 2:
                value += 0.14
                reasons.append("SENTENCE_SHIFT")
        value = round(min(1.0, value), 3)
        medium = cls.MEDIUM_THRESHOLD if medium_threshold is None else medium_threshold
        high = cls.HIGH_THRESHOLD if high_threshold is None else high_threshold
        level = "HIGH" if value >= high else "MEDIUM" if value >= medium else "LOW"
        return SemanticRiskResult(value, level, reasons, level != "LOW")
