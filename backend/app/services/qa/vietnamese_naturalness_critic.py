import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from app.services.translation.context_assembler import estimate_tokens


NATURALNESS_CRITIC_PROMPT_VERSION = "vietnamese-naturalness-critic-v1"
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

    def passed(self, threshold: float = NATURALNESS_PASS_THRESHOLD) -> bool:
        """Chỉ PASS khi provider và ngưỡng điểm đều đạt."""
        return (
            self.status == "PASS"
            and self.error is None
            and self.score is not None
            and self.score >= threshold
            and not self.issues
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
        }


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
    if status == "PASS" and (issues or "FAIL" in normalized_checks.values()):
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
        "previous_context": _normalize_previous_context(previous_context),
        "prompt_version": NATURALNESS_CRITIC_PROMPT_VERSION,
        "critic_model": model or "",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class VietnameseNaturalnessCritic:
    """Critic chỉ đánh giá tiếng Việt; không tự viết lại candidate."""

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
    ) -> VietnameseNaturalnessResult:
        selected_previous = _normalize_previous_context(previous_context)
        selected_glossary = dict(glossary_terms or {})
        selected_entities = dict(entity_context or {})
        signature = naturalness_signature(
            source_text, translated_text, document_type, register, sentence_style,
            selected_glossary, selected_entities, model, selected_previous,
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
                + estimate_tokens(str(document_type))
                + estimate_tokens(str(register))
                + estimate_tokens(str(sentence_style))
                + 480
            )

        request_size = request_tokens()
        if request_size > limit and selected_previous:
            selected_previous = []
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
                {}, message, request_size, 0.0, 0, signature,
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
            )
            result = validate_naturalness_result(raw)
            return VietnameseNaturalnessResult(
                result.status, result.score, result.issues, result.checks, result.error,
                request_size, (time.perf_counter() - started) * 1000, 1, signature,
            )
        except Exception as exc:
            message = str(exc)
            return VietnameseNaturalnessResult(
                "ERROR", None,
                [{
                    "type": "VIETNAMESE_NATURALNESS_ERROR",
                    "severity": "ERROR",
                    "source_span": "",
                    "target_span": "",
                    "message": message,
                }],
                {}, message, request_size, (time.perf_counter() - started) * 1000, 1, signature,
            )


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
