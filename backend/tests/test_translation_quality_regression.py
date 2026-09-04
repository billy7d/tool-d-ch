from types import SimpleNamespace

from app.services.translation.translation_eval_dataset import (
    DOMAINS,
    SCORE_FIELDS,
    load_official_dataset,
    validate_official_dataset,
)
from app.services.translation.translation_eval_metrics import compare_regression
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.vietnamese_editorial_assurance import TranslationPublicationAssuranceService


def test_official_dataset_has_required_size_domains_and_adversarial_metadata():
    rows = load_official_dataset()
    assert len(rows) >= 200
    assert {row["domain"] for row in rows} == set(DOMAINS)
    assert not validate_official_dataset(rows)
    assert all(all(field in row for field in SCORE_FIELDS) for row in rows)
    assert any("polysemy" in row["adversarial_tags"] for row in rows)
    assert any("multi_paragraph" in row["adversarial_tags"] for row in rows)
    assert all(row["evaluation_status"] == "PENDING_HUMAN" for row in rows if row.get("preferred_output"))


def test_regression_gate_accepts_allowed_naturalness_change_and_rejects_critical_regression():
    baseline = {
        "semantic_critical_error_rate": 0.01,
        "naturalness_pass_rate": 0.80,
        "hard_glossary_error_rate": 0.02,
        "median_provider_calls_per_node": 1.0,
    }
    candidate = {
        "semantic_critical_error_rate": 0.01,
        "naturalness_pass_rate": 0.78,
        "hard_glossary_error_rate": 0.02,
        "median_provider_calls_per_node": 1.0,
    }
    assert compare_regression(baseline, candidate, max_provider_calls_per_node=1.0)["passed"]

    critical = dict(candidate, semantic_critical_error_rate=0.02)
    result = compare_regression(baseline, critical, max_provider_calls_per_node=1.0)
    assert not result["passed"]
    assert not result["checks"]["semantic_critical_tolerance_zero"]["passed"]


def test_regression_gate_detects_glossary_and_provider_call_regressions():
    baseline = {
        "semantic_critical_error_rate": 0.0,
        "naturalness_pass_rate": 0.9,
        "hard_glossary_error_rate": 0.0,
        "median_provider_calls_per_node": 1.0,
    }
    candidate = {
        "semantic_critical_error_rate": 0.0,
        "naturalness_pass_rate": 0.9,
        "hard_glossary_error_rate": 0.01,
        "median_provider_calls_per_node": 2.0,
    }
    result = compare_regression(baseline, candidate, max_provider_calls_per_node=1.0)
    assert not result["passed"]
    assert not result["checks"]["hard_glossary_no_regression"]["passed"]
    assert not result["checks"]["provider_call_budget"]["passed"]


def test_quality_tiers_reuse_existing_qa_level_and_control_expensive_assurance():
    def config(qa_level: str):
        return TranslationConfig.from_project(SimpleNamespace(
            id="quality-tier", title="Quality tiers", selected_model="mock",
            document_type="GENERAL", translation_mode="NATURAL", qa_level=qa_level,
            style_guide={}, custom_instructions="",
        ))

    fast = config("FAST")
    balanced = config("BALANCED")
    publishing = config("PUBLISHING")

    assert fast.quality_tier == "FAST"
    assert balanced.quality_tier == "HIGH_QUALITY"
    assert publishing.quality_tier == "PUBLISHING"
    assert not TranslationPublicationAssuranceService.naturalness_enabled(SimpleNamespace(config=fast))
    assert TranslationPublicationAssuranceService.naturalness_enabled(SimpleNamespace(config=publishing))
