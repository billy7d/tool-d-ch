import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional

from app.services.qa.vietnamese_style_lint import STYLE_LINT_VERSION, StyleLintIssue, VietnameseStyleLint
from app.services.translation.context_assembler import estimate_tokens


# v2: critic nhận thêm deterministic_style_hints từ VietnameseStyleLint.
# v3: thêm sổ xưng hô và cho phép PASS kèm ghi chú INFO.
# v4: chuẩn hóa JSON lệch schema, lint WARNING lật PASS, JSON hỏng thì lint quyết định.
NATURALNESS_CRITIC_PROMPT_VERSION = "vietnamese-naturalness-critic-v4"
PAIRWISE_COMPARE_VERSION = "vietnamese-naturalness-pairwise-v1"
NATURALNESS_PASS_THRESHOLD = 0.80
NATURALNESS_REWRITE_THRESHOLD = 0.55

NATURALNESS_CHECKS = (
    "literal_calque",
    "word_order",
    "collocation",
    "cohesion",
    "pronoun_reference",
    "register",
    "redundancy",
    "nominalization",
    "passive_voice",
    "sentence_flow",
)

NATURALNESS_ISSUE_TYPES = {
    "LITERAL_CALQUE",
    "WORD_ORDER",
    "COLLOCATION",
    "COHESION",
    "PRONOUN_REFERENCE",
    "REGISTER",
    "REDUNDANCY",
    "NOMINALIZATION",
    "PASSIVE_VOICE",
    "SENTENCE_FLOW",
    "NATURALNESS_ERROR",
    "VIETNAMESE_NATURALNESS_ERROR",
}

_CHECK_TO_ISSUE = {
    "literal_calque": "LITERAL_CALQUE",
    "word_order": "WORD_ORDER",
    "collocation": "COLLOCATION",
    "cohesion": "COHESION",
    "pronoun_reference": "PRONOUN_REFERENCE",
    "register": "REGISTER",
    "redundancy": "REDUNDANCY",
    "nominalization": "NOMINALIZATION",
    "passive_voice": "PASSIVE_VOICE",
    "sentence_flow": "SENTENCE_FLOW",
}

_QA_ISSUE_TYPES = {
    "LITERAL_CALQUE": "VIETNAMESE_LITERAL_CALQUE",
    "WORD_ORDER": "VIETNAMESE_WORD_ORDER",
    "COLLOCATION": "VIETNAMESE_COLLOCATION",
    "COHESION": "VIETNAMESE_COHESION",
    "PRONOUN_REFERENCE": "VIETNAMESE_PRONOUN",
    "REGISTER": "VIETNAMESE_REGISTER",
    "REDUNDANCY": "VIETNAMESE_REDUNDANCY",
    "NOMINALIZATION": "VIETNAMESE_NOMINALIZATION",
    "PASSIVE_VOICE": "VIETNAMESE_PASSIVE",
    "SENTENCE_FLOW": "VIETNAMESE_SENTENCE_FLOW",
}


@dataclass(frozen=True)
class VietnameseNaturalnessResult:
    status: str
    score: Optional[float]
    issues: List[Dict[str, Any]] = field(default_factory=list)
    checks: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    critic_request_tokens: int = 0
    critic_latency_ms: float = 0.0
    critic_calls: int = 1
    signature: str = ""
    # Kết quả lint deterministic, kể cả INFO; WARNING của lint có thể lật PASS của critic (xem verdict_source).
    lint_issues: List[Dict[str, Any]] = field(default_factory=list)
    # CRITIC: theo critic; LINT_OVERRIDE: critic PASS nhưng lint có WARNING; LINT_FALLBACK: JSON critic không dùng được.
    verdict_source: str = "CRITIC"
    # Những chỗ JSON lệch schema đã được chuẩn hóa, để audit độ tin cậy của critic.
    schema_repairs: List[str] = field(default_factory=list)

    def passed(self, threshold: float = NATURALNESS_PASS_THRESHOLD) -> bool:
        """Chỉ PASS khi provider và ngưỡng điểm đều đạt; ghi chú INFO không chặn xuất bản."""
        return (
            self.status == "PASS"
            and self.error is None
            and self.score is not None
            and self.score >= threshold
            and not _blocking_issues(self.issues)
            and all(self.checks.get(key) == "PASS" for key in NATURALNESS_CHECKS)
        )

    def rewrite_allowed(
        self,
        rewrite_threshold: float = NATURALNESS_REWRITE_THRESHOLD,
        pass_threshold: float = NATURALNESS_PASS_THRESHOLD,
    ) -> bool:
        """Cho phép đúng vùng điểm trung gian; ERROR và điểm thấp không được sửa tự động."""
        return (
            self.status == "FAIL"
            and self.error is None
            and self.score is not None
            and rewrite_threshold <= self.score < pass_threshold
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "issues": self.issues,
            "checks": self.checks,
            "error": self.error,
            "critic_request_tokens": self.critic_request_tokens,
            "critic_latency_ms": self.critic_latency_ms,
            "critic_calls": self.critic_calls,
            "signature": self.signature,
            "lint_issues": self.lint_issues,
            "verdict_source": self.verdict_source,
            "schema_repairs": self.schema_repairs,
        }


def blocking_issues(issues: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """INFO là ghi chú tham khảo; chỉ WARNING/ERROR mới chặn PASS."""
    return [item for item in issues or [] if str(item.get("severity", "WARNING")).upper() != "INFO"]


_blocking_issues = blocking_issues


def naturalness_cache_key(source_text: str, translated_text: str, critic_model: str) -> str:
    """Khóa tái sử dụng kết luận đã duyệt: đổi chữ, đổi critic, lint hay model thì phải chấm lại."""
    payload = {
        "source": source_text or "",
        "translation": translated_text or "",
        "prompt_version": NATURALNESS_CRITIC_PROMPT_VERSION,
        "lint_version": STYLE_LINT_VERSION,
        "critic_model": critic_model or "",
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_previous_context(value: Any) -> Any:
    """Chuyển context song ngữ về JSON nhỏ, không đưa object SQLAlchemy vào prompt."""
    if value is None:
        return []
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        normalized = []
        for item in value:
            if isinstance(item, dict):
                normalized.append({str(key): str(part) for key, part in item.items()})
            else:
                normalized.append({
                    "source": str(getattr(item, "source", "")),
                    "translation": str(getattr(item, "translation", getattr(item, "target", ""))),
                })
        return normalized
    return str(value)


def _normalize_issue(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict) or not str(item.get("message", "")).strip():
        raise ValueError("Naturalness critic issue item không hợp lệ")
    issue_type = str(item.get("type", "NATURALNESS_ERROR")).strip().upper().replace(" ", "_")
    if issue_type not in NATURALNESS_ISSUE_TYPES:
        raise ValueError(f"Naturalness critic issue type không hợp lệ: {issue_type}")
    severity = str(item.get("severity", "WARNING")).upper()
    if severity not in {"INFO", "WARNING", "ERROR"}:
        raise ValueError("Naturalness critic severity không hợp lệ")
    return {
        "type": issue_type,
        "severity": severity,
        "source_span": str(item.get("source_span", "")),
        "target_span": str(item.get("target_span", "")),
        "message": str(item["message"]).strip(),
    }


_STATUS_ALIASES = {
    "PASS": "PASS", "PASSED": "PASS", "OK": "PASS", "GOOD": "PASS", "NATURAL": "PASS",
    "FAIL": "FAIL", "FAILED": "FAIL", "NEEDS_REVISION": "FAIL", "UNNATURAL": "FAIL",
    "ERROR": "ERROR",
}
_SEVERITY_ALIASES = {
    "INFO": "INFO", "LOW": "INFO", "MINOR": "INFO", "NOTE": "INFO", "SUGGESTION": "INFO", "TRIVIAL": "INFO",
    "WARNING": "WARNING", "WARN": "WARNING", "MEDIUM": "WARNING", "MODERATE": "WARNING",
    "ERROR": "ERROR", "HIGH": "ERROR", "MAJOR": "ERROR", "CRITICAL": "ERROR", "SEVERE": "ERROR",
}
_ISSUE_TYPE_ALIASES = {
    "CALQUE": "LITERAL_CALQUE", "LITERAL": "LITERAL_CALQUE", "LITERAL_TRANSLATION": "LITERAL_CALQUE",
    "PASSIVE": "PASSIVE_VOICE", "PRONOUN": "PRONOUN_REFERENCE", "REFERENCE": "PRONOUN_REFERENCE",
    "FLOW": "SENTENCE_FLOW", "FLUENCY": "SENTENCE_FLOW", "GRAMMAR": "SENTENCE_FLOW", "READABILITY": "SENTENCE_FLOW",
    "STYLE": "REGISTER", "TONE": "REGISTER", "FORMALITY": "REGISTER",
    "REDUNDANT": "REDUNDANCY", "WORDINESS": "REDUNDANCY", "VERBOSITY": "REDUNDANCY",
    "NOMINALISATION": "NOMINALIZATION", "ORDER": "WORD_ORDER", "SYNTAX": "WORD_ORDER",
    "WORD_CHOICE": "COLLOCATION", "LEXICAL": "COLLOCATION", "COHERENCE": "COHESION",
}
_PASS_VALUES = {"PASS", "PASSED", "OK", "TRUE", "YES", "GOOD"}
_FAIL_VALUES = {"FAIL", "FAILED", "FALSE", "NO", "BAD"}


def _repair_issue(item: Any, repairs: List[str]) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        if not item.strip():
            return None
        repairs.append("issue_string")
        return {"type": "SENTENCE_FLOW", "severity": "WARNING", "source_span": "", "target_span": "", "message": item.strip()}
    if not isinstance(item, dict):
        repairs.append("issue_dropped")
        return None
    lowered = {str(key).lower(): value for key, value in item.items()}
    raw_type = str(lowered.get("type") or lowered.get("category") or lowered.get("check") or "").strip().upper().replace(" ", "_").replace("-", "_")
    issue_type = _ISSUE_TYPE_ALIASES.get(raw_type, raw_type)
    if issue_type not in NATURALNESS_ISSUE_TYPES:
        repairs.append("issue_type")
        issue_type = "SENTENCE_FLOW"
    raw_severity = str(lowered.get("severity") or lowered.get("level") or "WARNING").strip().upper()
    severity = _SEVERITY_ALIASES.get(raw_severity)
    if severity is None:
        repairs.append("issue_severity")
        severity = "WARNING"
    elif severity != raw_severity:
        repairs.append("issue_severity")
    message = next(
        (str(lowered[key]).strip() for key in ("message", "description", "explanation", "reason", "comment")
         if str(lowered.get(key) or "").strip()),
        "",
    )
    target_span = str(lowered.get("target_span") or lowered.get("span") or lowered.get("text") or "")
    if not message:
        repairs.append("issue_message")
        message = f"{issue_type}: {target_span}".strip(": ") or issue_type
    return {
        "type": issue_type, "severity": severity,
        "source_span": str(lowered.get("source_span") or ""), "target_span": target_span, "message": message,
    }


def _repair_score(value: Any, repairs: List[str]) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        score = float(str(value).strip().rstrip("%"))
    except ValueError:
        return None
    if score > 1:
        # Model nhỏ hay chấm thang 10 hoặc 100.
        repairs.append("score_scale")
        score = score / 10 if score <= 10 else score / 100
    return min(1.0, max(0.0, score))


def normalize_naturalness_payload(raw: Any) -> tuple[Dict[str, Any], List[str]]:
    """Chuẩn hóa JSON lệch schema mà model 7B hay trả về, trước khi kiểm tra strict.

    Chỉ sửa hình thức (tên khóa, kiểu dữ liệu, thang điểm, alias); mọi mâu thuẫn về
    nội dung được giải theo hướng bảo thủ (PASS kèm lỗi chặn => FAIL). Payload rỗng
    hoặc FAIL không nêu lý do vẫn bị từ chối để rơi về lint.
    """
    repairs: List[str] = []
    if isinstance(raw, str):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("Naturalness critic không trả về JSON object")
        raw = json.loads(match.group(0))
        repairs.append("json_extracted")
    if not isinstance(raw, dict):
        raise ValueError("Naturalness critic phải trả về JSON object")
    data = {str(key).lower(): value for key, value in raw.items()}

    status = _STATUS_ALIASES.get(str(data.get("status") or data.get("verdict") or "").strip().upper())
    if status == "ERROR":
        raise ValueError(f"Naturalness critic tự báo ERROR: {data.get('error') or ''}".strip())

    raw_issues = data.get("issues")
    if raw_issues is None:
        raw_issues = []
    elif isinstance(raw_issues, (dict, str)):
        repairs.append("issues_not_list")
        raw_issues = [raw_issues] if not (isinstance(raw_issues, dict) and not raw_issues) else []
    elif not isinstance(raw_issues, list):
        raise ValueError("Naturalness critic issues không đọc được")
    issues = [issue for issue in (_repair_issue(item, repairs) for item in raw_issues) if issue]
    blocking_types = {item["type"] for item in _blocking_issues(issues)}

    raw_checks = data.get("checks")
    checks: Dict[str, str] = {}
    if isinstance(raw_checks, list):
        # Một số model trả danh sách tên check bị FAIL.
        repairs.append("checks_list")
        failed = {str(item).lower() for item in raw_checks}
        raw_checks = {key: ("FAIL" if key in failed else "PASS") for key in NATURALNESS_CHECKS}
    raw_checks = {str(key).lower(): value for key, value in (raw_checks or {}).items()} if isinstance(raw_checks, dict) else {}
    for key in NATURALNESS_CHECKS:
        value = raw_checks.get(key)
        if isinstance(value, dict):
            value = value.get("status", value.get("result"))
        if isinstance(value, bool):
            normalized = "PASS" if value else "FAIL"
        elif isinstance(value, (int, float)):
            normalized = "PASS" if float(value) >= 0.5 else "FAIL"
        else:
            text = str(value or "").strip().upper()
            normalized = "PASS" if text in _PASS_VALUES else "FAIL" if text in _FAIL_VALUES else ""
        if not normalized:
            normalized = "FAIL" if key.upper() in blocking_types else "PASS"
            repairs.append("check_derived")
        elif normalized != value:
            repairs.append("check_value")
        checks[key] = normalized
    failed_checks = [key for key, value in checks.items() if value == "FAIL"]

    if status is None:
        repairs.append("status_derived")
        status = "FAIL" if blocking_types or failed_checks else "PASS"
    if status == "PASS" and (blocking_types or failed_checks):
        repairs.append("pass_contradiction")
        status = "FAIL"
    if status == "FAIL" and not blocking_types and not failed_checks:
        raise ValueError("Naturalness critic FAIL nhưng không nêu lỗi nào")

    score = _repair_score(data.get("score"), repairs)
    if score is None:
        repairs.append("score_derived")
        score = max(0.0, round(1.0 - 0.15 * len(_blocking_issues(issues)), 2)) if status == "FAIL" else 0.9

    return {"status": status, "score": score, "issues": issues, "checks": checks}, sorted(set(repairs))


def validate_naturalness_result(raw: Any) -> VietnameseNaturalnessResult:
    """Kiểm tra strict JSON và từ chối schema mâu thuẫn trước khi orchestration sử dụng."""
    if not isinstance(raw, dict):
        raise ValueError("Naturalness critic phải trả về JSON object")
    status = str(raw.get("status", "")).upper()
    if status not in {"PASS", "FAIL", "ERROR"}:
        raise ValueError("Naturalness critic thiếu status PASS/FAIL/ERROR hợp lệ")
    if status == "ERROR":
        message = str(raw.get("error", "")).strip()
        if not message:
            raise ValueError("Naturalness critic ERROR phải có error")
        return VietnameseNaturalnessResult(
            "ERROR", None, [], {}, message,
            critic_calls=0,
        )

    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
        raise ValueError("Naturalness critic score phải nằm trong 0..1")
    checks = raw.get("checks")
    if not isinstance(checks, dict) or not set(NATURALNESS_CHECKS).issubset(checks):
        raise ValueError("Naturalness critic thiếu checks bắt buộc")
    normalized_checks = {key: str(checks[key]).upper() for key in NATURALNESS_CHECKS}
    if any(value not in {"PASS", "FAIL"} for value in normalized_checks.values()):
        raise ValueError("Naturalness critic checks chỉ nhận PASS/FAIL")
    raw_issues = raw.get("issues")
    if not isinstance(raw_issues, list):
        raise ValueError("Naturalness critic issues phải là danh sách")
    issues = [_normalize_issue(item) for item in raw_issues]
    if status == "PASS" and (_blocking_issues(issues) or "FAIL" in normalized_checks.values()):
        raise ValueError("Naturalness PASS mâu thuẫn với issues/checks")
    if status == "FAIL" and not issues and "FAIL" not in normalized_checks.values():
        raise ValueError("Naturalness FAIL phải có issue hoặc check FAIL")
    return VietnameseNaturalnessResult(status, float(score), issues, normalized_checks)


def naturalness_signature(
    source_text: str,
    translated_text: str,
    document_type: str,
    register: str,
    sentence_style: str,
    glossary_terms: Optional[Dict[str, str]],
    entity_context: Optional[Dict[str, str]],
    model: str,
    previous_context: Any = None,
    address_terms: Optional[Dict[str, Any]] = None,
) -> str:
    """Tạo identity ổn định để P1 có thể thêm cache mà không đổi API critic."""
    payload = {
        "source": source_text or "",
        "translation": translated_text or "",
        "document_type": str(document_type or "GENERAL").upper(),
        "register": str(register or "").upper(),
        "sentence_style": str(sentence_style or "").upper(),
        "glossary": dict(glossary_terms or {}),
        "entities": dict(entity_context or {}),
        "address_terms": dict(address_terms or {}),
        "previous_context": _normalize_previous_context(previous_context),
        "prompt_version": NATURALNESS_CRITIC_PROMPT_VERSION,
        "critic_model": model or "",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _error_result(
    message: str, request_size: int, latency: float, signature: str, lint_issues: List[Dict[str, Any]],
) -> VietnameseNaturalnessResult:
    return VietnameseNaturalnessResult(
        "ERROR", None,
        [{
            "type": "VIETNAMESE_NATURALNESS_ERROR",
            "severity": "ERROR",
            "source_span": "",
            "target_span": "",
            "message": message,
        }],
        {}, message, request_size, latency, 1, signature, lint_issues,
    )


def _lint_verdict(
    lint_issues: List[Dict[str, Any]],
    pass_threshold: float,
    rewrite_threshold: float,
    source: str,
    request_size: int,
    latency: float,
    signature: str,
    repairs: List[str],
    critic: Optional[VietnameseNaturalnessResult] = None,
) -> VietnameseNaturalnessResult:
    """Kết luận dựa trên lint; điểm FAIL đặt giữa vùng rewrite để bước biên tập được chạy."""
    warnings = [item for item in lint_issues if str(item.get("severity", "")).upper() == "WARNING"]
    checks = dict(critic.checks) if critic else {key: "PASS" for key in NATURALNESS_CHECKS}
    issues = list(critic.issues) if critic else []
    for hint in warnings:
        key = str(hint.get("type", "")).lower()
        if key in checks:
            checks[key] = "FAIL"
        issues.append(StyleLintIssue(**{
            name: str(hint.get(name, "")) for name in ("rule", "type", "severity", "target_span", "message", "suggestion")
        }).to_naturalness_issue())
    if warnings:
        midpoint = round((rewrite_threshold + pass_threshold) / 2, 3)
        score = min(critic.score, midpoint) if critic and critic.score is not None else midpoint
        status = "FAIL"
    else:
        score, status = pass_threshold, "PASS"
    return VietnameseNaturalnessResult(
        status, score, issues, checks, None, request_size, latency, 1, signature, lint_issues, source, list(repairs),
    )


class VietnameseNaturalnessCritic:
    """Critic chỉ đánh giá tiếng Việt; không tự viết lại candidate."""

    @classmethod
    def lint_review(
        cls,
        source_text: str,
        translated_text: str,
        document_type: str = "GENERAL",
        extra_hints: Optional[List[Dict[str, Any]]] = None,
        pass_threshold: float = NATURALNESS_PASS_THRESHOLD,
        rewrite_threshold: float = NATURALNESS_REWRITE_THRESHOLD,
    ) -> VietnameseNaturalnessResult:
        """Kết luận chỉ bằng lint, không gọi model.

        Đo thật trên qwen2.5:7b: mỗi lượt critic mất ~30s mà gần như mọi bản sát chữ nó bắt được
        thì lint cũng đã bắt; lint không báo nhầm bản tự nhiên nào trong bộ đánh giá.
        """
        lint_issues = [
            item.to_hint() for item in VietnameseStyleLint.lint(translated_text, source_text, document_type)
        ] + [dict(item) for item in extra_hints or []]
        result = _lint_verdict(lint_issues, pass_threshold, rewrite_threshold, "LINT_ONLY", 0, 0.0, "", [])
        return replace(result, critic_calls=0)

    @classmethod
    def review(
        cls,
        provider: Any,
        source_text: str,
        translated_text: str,
        document_type: str = "GENERAL",
        register: str = "",
        sentence_style: str = "MODERATE",
        previous_context: Any = None,
        glossary_terms: Optional[Dict[str, str]] = None,
        entity_context: Optional[Dict[str, str]] = None,
        model: str = "",
        address_terms: Optional[Dict[str, Any]] = None,
        extra_hints: Optional[List[Dict[str, Any]]] = None,
        pass_threshold: float = NATURALNESS_PASS_THRESHOLD,
        rewrite_threshold: float = NATURALNESS_REWRITE_THRESHOLD,
        lint_fallback: bool = True,
    ) -> VietnameseNaturalnessResult:
        selected_previous = _normalize_previous_context(previous_context)
        selected_glossary = dict(glossary_terms or {})
        selected_entities = dict(entity_context or {})
        # Sổ xưng hô là quyết định bắt buộc của project và rất nhỏ nên không bị cắt theo budget.
        selected_address = dict(address_terms or {})
        lint_issues = [
            item.to_hint() for item in VietnameseStyleLint.lint(translated_text, source_text, document_type)
        ] + [dict(item) for item in extra_hints or []]
        selected_hints = list(lint_issues)
        signature = naturalness_signature(
            source_text, translated_text, document_type, register, sentence_style,
            selected_glossary, selected_entities, model, selected_previous, selected_address,
        )

        try:
            limit = int(provider.effective_context_window(model))
        except Exception:
            limit = 8192

        def request_tokens() -> int:
            return (
                estimate_tokens(source_text or "")
                + estimate_tokens(translated_text or "")
                + estimate_tokens(json.dumps(selected_previous, ensure_ascii=False))
                + estimate_tokens(json.dumps(selected_glossary, ensure_ascii=False))
                + estimate_tokens(json.dumps(selected_entities, ensure_ascii=False))
                + estimate_tokens(json.dumps(selected_hints, ensure_ascii=False))
                + estimate_tokens(json.dumps(selected_address, ensure_ascii=False))
                + estimate_tokens(str(document_type))
                + estimate_tokens(str(register))
                + estimate_tokens(str(sentence_style))
                + 480
            )

        request_size = request_tokens()
        if request_size > limit and selected_previous:
            selected_previous = []
            request_size = request_tokens()
        if request_size > limit and selected_hints:
            selected_hints = []
            request_size = request_tokens()
        if request_size > limit and selected_entities:
            selected_entities = {}
            request_size = request_tokens()
        if request_size > limit and selected_glossary:
            selected_glossary = {}
            request_size = request_tokens()
        if request_size > limit:
            message = f"Naturalness critic request {request_size} token vượt context {limit}."
            return VietnameseNaturalnessResult(
                "ERROR", None,
                [{
                    "type": "VIETNAMESE_NATURALNESS_ERROR",
                    "severity": "ERROR",
                    "source_span": "",
                    "target_span": "",
                    "message": message,
                }],
                {}, message, request_size, 0.0, 0, signature, lint_issues,
            )

        started = time.perf_counter()
        try:
            raw = provider.review_naturalness(
                source_text=source_text,
                translated_text=translated_text,
                document_type=document_type,
                register=register,
                sentence_style=sentence_style,
                previous_context=selected_previous,
                glossary_terms=selected_glossary,
                entity_context=selected_entities,
                model=model,
                style_hints=selected_hints,
                address_terms=selected_address,
            )
            payload, repairs = normalize_naturalness_payload(raw)
            critic = validate_naturalness_result(payload)
        except ValueError as exc:
            # Model đã trả lời nhưng JSON không dùng được (kể cả JSONDecodeError): lint quyết định thay.
            latency = (time.perf_counter() - started) * 1000
            if lint_fallback:
                return _lint_verdict(
                    lint_issues, pass_threshold, rewrite_threshold, "LINT_FALLBACK",
                    request_size, latency, signature, [f"unusable: {exc}"],
                )
            return _error_result(str(exc), request_size, latency, signature, lint_issues)
        except Exception as exc:
            # Lỗi hạ tầng (timeout, mất kết nối) vẫn fail-closed: không biết model có chạy đúng không.
            return _error_result(str(exc), request_size, (time.perf_counter() - started) * 1000, signature, lint_issues)

        latency = (time.perf_counter() - started) * 1000
        lint_warnings = [item for item in lint_issues if str(item.get("severity", "")).upper() == "WARNING"]
        if critic.status == "PASS" and lint_warnings:
            # Đo thật trên qwen2.5:7b: critic cho PASS 1.0 cả bản sát chữ, còn lint không báo nhầm bản tự nhiên.
            return _lint_verdict(
                lint_issues, pass_threshold, rewrite_threshold, "LINT_OVERRIDE",
                request_size, latency, signature, repairs, critic,
            )
        return VietnameseNaturalnessResult(
            critic.status, critic.score, critic.issues, critic.checks, critic.error,
            request_size, latency, 1, signature, lint_issues, "CRITIC", repairs,
        )


@dataclass(frozen=True)
class PairwiseNaturalnessResult:
    # True: bản gốc thắng ở cả hai thứ tự; False: không thắng; None: provider không hỗ trợ/lỗi.
    original_preferred: Optional[bool]
    calls: int = 0
    latency_ms: float = 0.0
    reasons: List[str] = field(default_factory=list)


def _pairwise_choice(raw: Any) -> str:
    if not isinstance(raw, dict):
        raise ValueError("Pairwise naturalness phải trả về JSON object")
    choice = str(raw.get("preferred", "")).strip().upper()
    if choice not in {"A", "B", "TIE"}:
        raise ValueError("Pairwise naturalness preferred chỉ nhận A/B/TIE")
    return choice


def compare_original_and_rewrite(
    provider: Any,
    source_text: str,
    original: str,
    rewritten: str,
    document_type: str = "GENERAL",
    register: str = "",
    model: str = "",
) -> PairwiseNaturalnessResult:
    """Hỏi hai lần với thứ tự đảo để khử thiên lệch vị trí; chỉ veto khi bản gốc thắng cả hai lần."""
    started = time.perf_counter()
    calls = 0
    reasons: List[str] = []
    original_wins = 0
    try:
        for original_slot, candidate_a, candidate_b in (("A", original, rewritten), ("B", rewritten, original)):
            calls += 1
            raw = provider.compare_naturalness(
                source_text=source_text, candidate_a=candidate_a, candidate_b=candidate_b,
                document_type=document_type, register=register, model=model,
            )
            choice = _pairwise_choice(raw)
            reasons.append(str(raw.get("reason", "")))
            if choice == original_slot:
                original_wins += 1
    except Exception:
        # Provider cũ không có pairwise hoặc trả sai schema: không veto, các gate khác vẫn giữ nguyên.
        return PairwiseNaturalnessResult(None, calls, (time.perf_counter() - started) * 1000, reasons)
    return PairwiseNaturalnessResult(original_wins == 2, calls, (time.perf_counter() - started) * 1000, reasons)


def naturalness_issue_to_qa_type(issue_type: str) -> str:
    """Đổi taxonomy critic sang issue type mà QA Editor hiện có thể hiển thị."""
    normalized = str(issue_type or "NATURALNESS_ERROR").upper()
    return _QA_ISSUE_TYPES.get(normalized, "VIETNAMESE_NATURALNESS_ERROR")


def naturalness_result_to_qa_issues(
    result: VietnameseNaturalnessResult,
    source_text: str,
    translated_text: str,
) -> List[Dict[str, Any]]:
    """Chuẩn hóa issue critic thành payload QAIssueModel, không sửa nội dung tự động."""
    issues = list(result.issues)
    if not result.passed() and not issues:
        issues = [{
            "type": "VIETNAMESE_NATURALNESS_ERROR",
            "severity": "ERROR" if result.status == "ERROR" else "WARNING",
            "target_span": "",
            "message": result.error or "Bản dịch chưa đạt ngưỡng tự nhiên tiếng Việt.",
        }]
    return [{
        "issue_type": naturalness_issue_to_qa_type(item.get("type", "NATURALNESS_ERROR")),
        "severity": item.get("severity", "WARNING"),
        "message": item.get("message", "Naturalness critic phát hiện vấn đề."),
        "source_snippet": (source_text or "")[:150],
        "translation_snippet": (translated_text or "")[:150],
        "suggested_fix": "Chỉ biên tập lại cách diễn đạt rồi chạy lại Quality Gate và Semantic Assurance.",
    } for item in issues]
