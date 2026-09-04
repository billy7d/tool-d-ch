from app.services.translation.translation_eval_metrics import summarize_ab_preferences, summarize_evaluations


def test_metrics_keep_five_human_dimensions_and_runtime_telemetry_separate():
    rows = [
        {
            "semantic_accuracy": 5, "naturalness": 4, "terminology": 5,
            "contextual_coherence": 4, "register": 4,
            "naturalness_critic_calls": 1, "semantic_repairs": 0,
            "provider_calls": 2, "provider_calls_per_node": 2,
            "latency_ms": 100, "request_tokens": 80,
            "style_memory_hits": 1, "few_shot_count": 3,
            "hard_glossary_error": False, "semantic_critical_error": False,
            "deterministic_error": False, "rewrite_applied": True,
        },
        {
            "semantic_accuracy": 4, "naturalness": 3, "terminology": 4,
            "contextual_coherence": 5, "register": 3,
            "naturalness_critic_calls": 1, "semantic_repairs": 1,
            "provider_calls": 1, "provider_calls_per_node": 1,
            "latency_ms": 200, "request_tokens": 120,
            "style_memory_hits": 0, "few_shot_count": 0,
            "hard_glossary_error": False, "semantic_critical_error": False,
            "deterministic_error": True, "rewrite_applied": False,
        },
    ]
    metrics = summarize_evaluations(rows)

    assert metrics["samples"] == 2
    assert metrics["semantic_accuracy_mean"] == 4.5
    assert metrics["naturalness_mean"] == 3.5
    assert metrics["naturalness_score"] == 3.5
    assert metrics["naturalness_pass_rate"] == 0.5
    assert metrics["style_memory_hit_rate"] == 0.5
    assert metrics["few_shot_usage_rate"] == 0.5
    assert metrics["naturalness_calls"] == 2
    assert metrics["semantic_repairs"] == 1
    assert metrics["provider_calls"] == 3
    assert metrics["median_provider_calls_per_node"] == 1.5
    assert metrics["median_latency_ms"] == 150.0
    assert metrics["deterministic_error_rate"] == 0.5
    assert metrics["rated_samples"] == 2


def test_metrics_accept_pending_human_rows_without_fabricating_scores():
    metrics = summarize_evaluations([
        {"sample_id": "pending", "naturalness": None, "provider_calls": 1},
    ])

    assert metrics["naturalness_mean"] is None
    assert metrics["rated_samples"] == 0
    assert metrics["provider_calls"] == 1


def test_blind_ab_metrics_map_candidate_and_baseline_without_exposing_identity():
    metrics = summarize_ab_preferences([
        {"preference": "A", "candidate_side": "A"},
        {"preference": "B", "candidate_side": "A"},
        {"preference": "TIE", "candidate_side": "B"},
    ])

    assert metrics == {
        "rated_samples": 3,
        "candidate_preference_rate": 0.3333,
        "tie_rate": 0.3333,
        "baseline_preference_rate": 0.3333,
    }
