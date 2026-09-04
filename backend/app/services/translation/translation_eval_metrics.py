"""Tính metric và regression gate cho benchmark P1/P2."""

from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCORE_FIELDS = (
    "semantic_accuracy", "naturalness", "terminology",
    "contextual_coherence", "register",
)
CRITICAL_ERROR_FIELDS = (
    "semantic_critical_errors", "semantic_errors", "critical_errors",
)


def _numbers(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 1 <= float(value) <= 5:
            values.append(float(value))
    return values


def _mean(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    return round(sum(items) / len(items), 4) if items else None


def _sum_numeric(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float:
    total = 0.0
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total += float(value)
                break
    return total


def _rate(rows: Sequence[Mapping[str, Any]], predicate) -> float:
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4) if rows else 0.0


def summarize_evaluations(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Tổng hợp human score, gate lỗi và telemetry mà không tự chấm bản dịch."""
    items = list(rows)
    metrics: Dict[str, Any] = {"samples": len(items)}
    for field_name in SCORE_FIELDS:
        metrics[f"{field_name}_mean"] = _mean(_numbers(items, field_name))
    metrics.update({
        "naturalness_score": metrics["naturalness_mean"],
        "naturalness_pass_rate": _rate(
            items, lambda row: isinstance(row.get("naturalness"), (int, float)) and row["naturalness"] >= 4,
        ),
        "rewrite_rate": _rate(
            items, lambda row: bool(row.get("editorial_rewrite_count") or row.get("rewrite_applied")),
        ),
        "style_memory_hit_rate": _rate(
            items, lambda row: bool(row.get("style_memory_hit") or row.get("style_memory_hits")),
        ),
        "few_shot_usage_rate": _rate(
            items, lambda row: bool(row.get("few_shot_used") or row.get("few_shot_count")),
        ),
        "naturalness_calls": int(_sum_numeric(items, ("naturalness_critic_calls", "naturalness_calls"))),
        "semantic_repairs": int(_sum_numeric(items, ("semantic_repairs", "repair_attempts"))),
        "provider_calls": int(_sum_numeric(items, ("provider_calls", "provider_call_count"))),
        "latency_ms_total": round(_sum_numeric(items, ("latency_ms",)), 3),
        "request_tokens_total": int(_sum_numeric(items, ("request_tokens", "tokens"))),
    })
    latencies = [float(row["latency_ms"]) for row in items if isinstance(row.get("latency_ms"), (int, float))]
    calls = [float(row["provider_calls_per_node"]) for row in items if isinstance(row.get("provider_calls_per_node"), (int, float))]
    metrics["median_latency_ms"] = round(float(median(latencies)), 3) if latencies else None
    metrics["median_provider_calls_per_node"] = round(float(median(calls)), 4) if calls else None
    metrics["avg_provider_calls_per_node"] = round(metrics["provider_calls"] / len(items), 4) if items else 0.0
    metrics["semantic_critical_error_rate"] = _rate(
        items,
        lambda row: bool(
            row.get("semantic_critical_error")
            or row.get("semantic_critical_errors")
            or row.get("critical_error")
        ),
    )
    metrics["hard_glossary_error_rate"] = _rate(
        items,
        lambda row: bool(row.get("hard_glossary_error") or row.get("glossary_error")),
    )
    metrics["deterministic_error_rate"] = _rate(
        items,
        lambda row: bool(row.get("deterministic_error") or row.get("quality_gate_error")),
    )
    metrics["rated_samples"] = sum(
        1 for row in items if all(isinstance(row.get(field_name), (int, float)) for field_name in SCORE_FIELDS)
    )
    return metrics


def summarize_ab_preferences(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Tổng hợp lựa chọn A/B mà không để lộ bản nào là candidate."""
    valid = []
    for row in rows:
        preference = str(
            row.get("preference")
            or row.get("preferred_side")
            or row.get("ab_preference")
            or ""
        ).strip().upper()
        if preference in {"DRAW", "TIE"}:
            preference = "TIE"
        if preference not in {"A", "B", "TIE"}:
            continue
        candidate_side = str(row.get("candidate_side") or "").strip().upper()
        if not candidate_side and isinstance(row.get("candidate_is_a"), bool):
            candidate_side = "A" if row["candidate_is_a"] else "B"
        if candidate_side not in {"A", "B"}:
            candidate_side = ""
        valid.append((preference, candidate_side))

    denominator = len(valid)
    candidate_wins = sum(1 for preference, side in valid if side and preference == side)
    baseline_wins = sum(1 for preference, side in valid if side and preference in {"A", "B"} and preference != side)
    ties = sum(1 for preference, _side in valid if preference == "TIE")
    return {
        "rated_samples": denominator,
        "candidate_preference_rate": round(candidate_wins / denominator, 4) if denominator else 0.0,
        "tie_rate": round(ties / denominator, 4) if denominator else 0.0,
        "baseline_preference_rate": round(baseline_wins / denominator, 4) if denominator else 0.0,
    }


def summarize_ab_evaluations(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Alias mô tả rõ hơn cho caller benchmark/human review."""
    return summarize_ab_preferences(rows)


def compare_regression(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    naturalness_tolerance: float = 0.02,
    max_provider_calls_per_node: Optional[float] = None,
) -> Dict[str, Any]:
    """Regression gate: semantic critical 0, naturalness tối đa giảm 2 điểm phần trăm."""
    baseline_semantic = float(baseline.get("semantic_critical_error_rate", 0.0) or 0.0)
    candidate_semantic = float(candidate.get("semantic_critical_error_rate", 0.0) or 0.0)
    baseline_naturalness = float(baseline.get("naturalness_pass_rate", 0.0) or 0.0)
    candidate_naturalness = float(candidate.get("naturalness_pass_rate", 0.0) or 0.0)
    baseline_glossary = float(baseline.get("hard_glossary_error_rate", 0.0) or 0.0)
    candidate_glossary = float(candidate.get("hard_glossary_error_rate", 0.0) or 0.0)
    configured_limit = max_provider_calls_per_node
    if configured_limit is None:
        configured_limit = baseline.get("configured_max_provider_calls_per_node")
    candidate_calls = candidate.get("median_provider_calls_per_node")
    checks = {
        "semantic_critical_tolerance_zero": {
            "passed": candidate_semantic <= baseline_semantic,
            "baseline": baseline_semantic,
            "candidate": candidate_semantic,
            "tolerance": 0.0,
        },
        "naturalness_two_percentage_points": {
            "passed": candidate_naturalness >= baseline_naturalness - naturalness_tolerance,
            "baseline": baseline_naturalness,
            "candidate": candidate_naturalness,
            "tolerance": naturalness_tolerance,
        },
        "hard_glossary_no_regression": {
            "passed": candidate_glossary <= baseline_glossary,
            "baseline": baseline_glossary,
            "candidate": candidate_glossary,
            "tolerance": 0.0,
        },
        "provider_call_budget": {
            "passed": configured_limit is None or candidate_calls is None or float(candidate_calls) <= float(configured_limit),
            "baseline": baseline.get("median_provider_calls_per_node"),
            "candidate": candidate_calls,
            "configured_limit": configured_limit,
        },
    }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def regression_gate(*args, **kwargs) -> Dict[str, Any]:
    """Tên alias ngắn cho caller CI hiện có."""
    return compare_regression(*args, **kwargs)
