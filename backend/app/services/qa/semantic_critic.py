import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.services.qa.semantic_issue_types import normalize_semantic_issue_type, semantic_severity
from app.services.translation.context_assembler import estimate_tokens


SEMANTIC_CRITIC_PROMPT_VERSION = "semantic-critic-v1"
_CHECKS = {"completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference"}


@dataclass(frozen=True)
class SemanticCriticResult:
    status: str
    score: Optional[float]
    errors: list[Dict[str, Any]] = field(default_factory=list)
    checks: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    critic_request_tokens: int = 0
    critic_latency_ms: float = 0.0
    critic_calls: int = 1


def semantic_signature(source_text: str, translated_text: str, glossary: Dict[str, str], entities: Dict[str, str], model: str, document_type: str) -> str:
    payload = {
        "source": source_text, "translation": translated_text, "glossary": glossary,
        "entities": entities, "prompt_version": SEMANTIC_CRITIC_PROMPT_VERSION,
        "model": model, "document_type": document_type,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def validate_semantic_result(raw: Any) -> SemanticCriticResult:
    if not isinstance(raw, dict):
        raise ValueError("Semantic critic phải trả về JSON object")
    status = str(raw.get("status", "")).upper()
    if status not in {"PASS", "FAIL"}:
        raise ValueError("Semantic critic thiếu status PASS/FAIL hợp lệ")
    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
        raise ValueError("Semantic critic score phải nằm trong 0..1")
    checks = raw.get("checks")
    if not isinstance(checks, dict) or not _CHECKS.issubset(checks):
        raise ValueError("Semantic critic thiếu checks bắt buộc")
    normalized_checks = {key: str(checks[key]).upper() for key in _CHECKS}
    if any(value not in {"PASS", "FAIL"} for value in normalized_checks.values()):
        raise ValueError("Semantic critic checks chỉ nhận PASS/FAIL")
    errors = raw.get("errors")
    if not isinstance(errors, list):
        raise ValueError("Semantic critic errors phải là danh sách")
    normalized_errors = []
    for item in errors:
        if not isinstance(item, dict) or not str(item.get("message", "")).strip():
            raise ValueError("Semantic critic error item không hợp lệ")
        issue_type = normalize_semantic_issue_type(str(item.get("type", "")))
        normalized_errors.append({
            "type": issue_type,
            "severity": semantic_severity(issue_type),
            "source_span": str(item.get("source_span", "")),
            "target_span": str(item.get("target_span", "")),
            "message": str(item["message"]).strip(),
        })
    if status == "PASS" and (normalized_errors or "FAIL" in normalized_checks.values()):
        raise ValueError("Semantic PASS mâu thuẫn với errors/checks")
    if status == "FAIL" and not normalized_errors:
        raise ValueError("Semantic FAIL phải có ít nhất một error")
    return SemanticCriticResult(status, float(score), normalized_errors, normalized_checks)


class SemanticCritic:
    @classmethod
    def review(cls, provider, source_text: str, translated_text: str, glossary: Dict[str, str], entities: Dict[str, str], document_type: str, model: str) -> SemanticCriticResult:
        selected_glossary = dict(glossary or {})
        selected_entities = dict(entities or {})
        limit = provider.effective_context_window(model)
        def token_estimate() -> int:
            return estimate_tokens(source_text) + estimate_tokens(translated_text) + estimate_tokens(json.dumps(selected_glossary, ensure_ascii=False)) + estimate_tokens(json.dumps(selected_entities, ensure_ascii=False)) + 320
        request_tokens = token_estimate()
        if request_tokens > limit and selected_entities:
            selected_entities = {}
            request_tokens = token_estimate()
        if request_tokens > limit and selected_glossary:
            selected_glossary = {}
            request_tokens = token_estimate()
        if request_tokens > limit:
            message = f"Semantic critic request {request_tokens} token vượt context {limit}."
            return SemanticCriticResult(
                "ERROR", None,
                [{"type": "SEMANTIC_QA_ERROR", "severity": "ERROR", "source_span": "", "target_span": "", "message": message}],
                {}, message, request_tokens, 0.0, 0,
            )
        started = time.perf_counter()
        try:
            raw = provider.review_semantic_fidelity(
                source_text, translated_text, selected_glossary, selected_entities, document_type, model=model,
            )
            result = validate_semantic_result(raw)
            return SemanticCriticResult(
                result.status, result.score, result.errors, result.checks,
                critic_request_tokens=request_tokens,
                critic_latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return SemanticCriticResult(
                "ERROR", None,
                [{"type": "SEMANTIC_QA_ERROR", "severity": "ERROR", "source_span": "", "target_span": "", "message": str(exc)}],
                {}, str(exc), request_tokens, (time.perf_counter() - started) * 1000,
            )
