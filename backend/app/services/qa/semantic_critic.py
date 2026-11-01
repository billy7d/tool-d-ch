import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.qa.semantic_issue_types import normalize_semantic_issue_type, semantic_severity
from app.services.translation.context_assembler import estimate_tokens
from app.services.translation.semantic_risk import SEMANTIC_POLICY_VERSION, SemanticRiskScorer


# v2: chuẩn hóa JSON lệch schema (description/true/"Partial") trước khi kiểm tra strict.
SEMANTIC_CRITIC_PROMPT_VERSION = "semantic-critic-v2"
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
    # Những chỗ JSON lệch schema đã được chuẩn hóa, để audit độ tin cậy của critic.
    schema_repairs: List[str] = field(default_factory=list)


def semantic_signature(
    source_text: str,
    translated_text: str,
    glossary: Dict[str, str],
    entities: Dict[str, str],
    model: str,
    document_type: str,
    policy: Optional[Dict[str, Any]] = None,
) -> str:
    selected_policy = dict(policy or {
        "version": SEMANTIC_POLICY_VERSION,
        "risk_medium": SemanticRiskScorer.MEDIUM_THRESHOLD,
        "risk_high": SemanticRiskScorer.HIGH_THRESHOLD,
        "max_repairs": 2,
    })
    payload = {
        "source": source_text, "translation": translated_text, "glossary": glossary,
        "entities": entities, "prompt_version": SEMANTIC_CRITIC_PROMPT_VERSION,
        "model": model, "document_type": document_type, "policy": selected_policy,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


_CHECK_ERROR_TYPES = {
    "completeness": "SEMANTIC_OMISSION",
    "meaning": "MEANING_DRIFT",
    "polarity": "MEANING_DRIFT",
    "modality": "MODALITY_ERROR",
    "causality": "CAUSALITY_ERROR",
    "scope": "SCOPE_ERROR",
    "entity_reference": "ENTITY_REFERENCE_ERROR",
}
_ERROR_TYPE_CHECKS = {
    "SEMANTIC_OMISSION": "completeness", "SEMANTIC_ADDITION": "completeness", "MEANING_DRIFT": "meaning",
    "MODALITY_ERROR": "modality", "CAUSALITY_ERROR": "causality", "SCOPE_ERROR": "scope",
    "CONDITION_ERROR": "scope", "COMPARISON_ERROR": "scope",
    "ENTITY_REFERENCE_ERROR": "entity_reference", "PRONOUN_AMBIGUITY": "entity_reference",
}
MIN_REPORTED_CHECKS_FOR_PASS = 5
_SEMANTIC_PASS = {"PASS", "PASSED", "OK", "TRUE", "YES", "CORRECT"}
# "Partial", "Mostly"... là không chắc chắn; với kiểm tra ngữ nghĩa thì coi như FAIL.
_SEMANTIC_FAIL = {"FAIL", "FAILED", "FALSE", "NO", "INCORRECT", "PARTIAL", "PARTIALLY", "MOSTLY", "UNCLEAR"}


def normalize_semantic_payload(raw: Any) -> tuple[Dict[str, Any], List[str]]:
    """Chuẩn hóa JSON lệch schema của model nhỏ trước khi kiểm tra strict.

    Đo trên qwen2.5:7b: 81% lượt ERROR chỉ vì dùng "description" thay "message" hoặc trả checks
    dạng true/"Pass"/"Partial", trong khi phán đoán nội dung phần lớn đúng. Mọi chỗ mơ hồ đều
    nghiêng về FAIL; payload không suy ra được kết luận vẫn bị từ chối (fail-closed).
    """
    repairs: List[str] = []
    if isinstance(raw, str):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("Semantic critic không trả về JSON object")
        raw = json.loads(match.group(0))
        repairs.append("json_extracted")
    if not isinstance(raw, dict):
        raise ValueError("Semantic critic phải trả về JSON object")
    data = {str(key).lower(): value for key, value in raw.items()}

    raw_errors = data.get("errors", data.get("issues"))
    if raw_errors is None:
        raw_errors = []
    elif isinstance(raw_errors, (dict, str)):
        repairs.append("errors_not_list")
        raw_errors = [raw_errors] if raw_errors else []
    elif not isinstance(raw_errors, list):
        raise ValueError("Semantic critic errors không đọc được")
    errors: List[Dict[str, Any]] = []
    for item in raw_errors:
        if isinstance(item, str):
            if item.strip():
                repairs.append("error_string")
                errors.append({"type": "MEANING_DRIFT", "message": item.strip()})
            continue
        if not isinstance(item, dict):
            repairs.append("error_dropped")
            continue
        lowered = {str(key).lower(): value for key, value in item.items()}
        message = next(
            (str(lowered[key]).strip() for key in ("message", "description", "explanation", "reason", "details", "comment")
             if str(lowered.get(key) or "").strip()),
            "",
        )
        if "message" not in lowered:
            repairs.append("error_message")
        issue_type = str(lowered.get("type") or lowered.get("category") or "MEANING_DRIFT")
        errors.append({
            "type": issue_type,
            "message": message or issue_type,
            "source_span": str(lowered.get("source_span") or ""),
            "target_span": str(lowered.get("target_span") or lowered.get("span") or ""),
        })

    raw_checks = data.get("checks")
    raw_checks = {str(key).lower(): value for key, value in raw_checks.items()} if isinstance(raw_checks, dict) else {}
    error_checks = {_ERROR_TYPE_CHECKS.get(normalize_semantic_issue_type(item["type"]), "meaning") for item in errors}
    checks: Dict[str, str] = {}
    for key in sorted(_CHECKS):
        value = raw_checks.get(key)
        if isinstance(value, dict):
            value = value.get("status", value.get("result"))
        if isinstance(value, bool):
            normalized = "PASS" if value else "FAIL"
        elif isinstance(value, (int, float)):
            normalized = "PASS" if float(value) >= 0.9 else "FAIL"
        else:
            text = str(value or "").strip().upper()
            normalized = "PASS" if text in _SEMANTIC_PASS else "FAIL" if text in _SEMANTIC_FAIL else ""
        if not normalized:
            normalized = "FAIL" if key in error_checks else "PASS"
            repairs.append("check_derived")
        elif normalized != value:
            repairs.append("check_value")
        checks[key] = normalized

    failed_checks = [key for key, value in checks.items() if value == "FAIL"]
    reported_checks = len(_CHECKS & set(raw_checks))
    status = str(data.get("status") or data.get("verdict") or "").strip().upper()
    status = {"PASSED": "PASS", "OK": "PASS", "FAILED": "FAIL"}.get(status, status)
    if status not in {"PASS", "FAIL"}:
        repairs.append("status_derived")
        status = "FAIL" if errors or failed_checks else "PASS"
    if status == "PASS" and (errors or failed_checks):
        repairs.append("pass_contradiction")
        status = "FAIL"
    if status == "PASS" and reported_checks < MIN_REPORTED_CHECKS_FOR_PASS:
        # PASS trần (không kèm checks) là quá ít bằng chứng để duyệt nghĩa.
        raise ValueError(f"Semantic critic PASS chỉ báo {reported_checks}/{len(_CHECKS)} checks")
    if status == "FAIL" and not errors:
        if not failed_checks:
            raise ValueError("Semantic critic FAIL nhưng không nêu lỗi nào")
        # Check FAIL mà không có error: dựng error từ chính check để repair biết sửa gì.
        repairs.append("error_from_check")
        errors = [
            {"type": _CHECK_ERROR_TYPES[key], "message": f"Critic đánh dấu check '{key}' không đạt.", "source_span": "", "target_span": ""}
            for key in failed_checks
        ]

    score = data.get("score")
    try:
        score = None if isinstance(score, bool) or score is None else float(str(score).strip().rstrip("%"))
    except ValueError:
        score = None
    if score is not None and score > 1:
        repairs.append("score_scale")
        score = score / 10 if score <= 10 else score / 100
    if score is None:
        repairs.append("score_derived")
        score = 1.0 if status == "PASS" else 0.4
    score = min(1.0, max(0.0, score))
    if status == "FAIL" and score >= 0.9:
        # Model hay cho FAIL kèm score 1.0; giữ kết luận FAIL, chỉ hạ điểm cho nhất quán.
        repairs.append("score_capped")
        score = 0.6
    return {"status": status, "score": score, "errors": errors, "checks": checks}, sorted(set(repairs))


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
            payload, repairs = normalize_semantic_payload(raw)
            result = validate_semantic_result(payload)
            return SemanticCriticResult(
                result.status, result.score, result.errors, result.checks,
                critic_request_tokens=request_tokens,
                critic_latency_ms=(time.perf_counter() - started) * 1000,
                schema_repairs=repairs,
            )
        except Exception as exc:
            return SemanticCriticResult(
                "ERROR", None,
                [{"type": "SEMANTIC_QA_ERROR", "severity": "ERROR", "source_span": "", "target_span": "", "message": str(exc)}],
                {}, str(exc), request_tokens, (time.perf_counter() - started) * 1000,
            )
