from types import SimpleNamespace

from app.services.qa.vietnamese_naturalness_critic import (
    NATURALNESS_CHECKS,
    VietnameseNaturalnessCritic,
    naturalness_result_to_qa_issues,
    validate_naturalness_result,
)
from app.services.translation.mock_provider import MockProvider


def _checks(value="PASS"):
    return {key: value for key in NATURALNESS_CHECKS}


def test_naturalness_critic_accepts_natural_vietnamese():
    result = VietnameseNaturalnessCritic.review(
        MockProvider(),
        "The company decided to take action immediately.",
        "Công ty quyết định hành động ngay.",
        document_type="GENERAL",
        register="NEUTRAL",
        sentence_style="MODERATE",
        model="mock",
    )

    assert result.status == "PASS"
    assert result.passed()
    assert result.score == 0.96
    assert result.issues == []
    assert set(result.checks) == set(NATURALNESS_CHECKS)


def test_naturalness_critic_detects_literal_calque_and_redundancy():
    result = VietnameseNaturalnessCritic.review(
        MockProvider(),
        "The company decided to take action immediately.",
        "Công ty đã đưa ra quyết định để thực hiện hành động ngay lập tức.",
        document_type="BUSINESS",
        register="PROFESSIONAL",
        sentence_style="MODERATE",
        model="mock",
    )

    assert result.status == "FAIL"
    assert not result.passed()
    assert result.rewrite_allowed()
    assert {item["type"] for item in result.issues} >= {"COLLOCATION", "LITERAL_CALQUE", "REDUNDANCY"}


def test_naturalness_critic_respects_domain_context_for_passive_example():
    provider = MockProvider()
    literal = VietnameseNaturalnessCritic.review(
        provider,
        "The proposal was approved by the board.",
        "Đề xuất đã được phê duyệt bởi hội đồng quản trị.",
        document_type="GENERAL",
        register="NEUTRAL",
        sentence_style="MODERATE",
        model="mock",
    )
    natural = VietnameseNaturalnessCritic.review(
        provider,
        "The proposal was approved by the board.",
        "Hội đồng quản trị đã phê duyệt đề xuất.",
        document_type="GENERAL",
        register="NEUTRAL",
        sentence_style="MODERATE",
        model="mock",
    )

    assert literal.status == "FAIL"
    assert natural.status == "PASS"
    assert natural.score > literal.score


def test_naturalness_validator_rejects_malformed_or_contradictory_json():
    try:
        validate_naturalness_result({"status": "PASS", "score": 0.9, "issues": [], "checks": {}})
        assert False, "Schema thiếu checks phải bị từ chối"
    except ValueError:
        pass

    try:
        validate_naturalness_result({
            "status": "PASS", "score": 0.9,
            "issues": [{"type": "LITERAL_CALQUE", "message": "Không tự nhiên"}],
            "checks": _checks(),
        })
        assert False, "PASS có issue phải bị từ chối"
    except ValueError:
        pass


def test_naturalness_provider_error_is_fail_closed():
    class BadProvider(MockProvider):
        def review_naturalness(self, *args, **kwargs):
            raise RuntimeError("provider timeout")

    result = VietnameseNaturalnessCritic.review(
        BadProvider(), "A source.", "Một nguồn.", model="mock",
    )

    assert result.status == "ERROR"
    assert result.score is None
    assert result.passed() is False
    assert result.issues[0]["type"] == "VIETNAMESE_NATURALNESS_ERROR"
    assert "provider timeout" in result.error


def test_naturalness_issue_taxonomy_uses_qa_editor_names():
    result = VietnameseNaturalnessCritic.review(
        MockProvider(),
        "The proposal was approved by the board.",
        "Đề xuất đã được phê duyệt bởi hội đồng quản trị.",
        model="mock",
    )

    issue_types = {item["issue_type"] for item in naturalness_result_to_qa_issues(result, "source", "target")}
    assert "VIETNAMESE_PASSIVE" in issue_types


def test_naturalness_context_overflow_does_not_call_provider():
    class TinyContextProvider(MockProvider):
        def __init__(self):
            self.naturalness_calls = 0

        def effective_context_window(self, model):
            return 20

        def review_naturalness(self, *args, **kwargs):
            self.naturalness_calls += 1
            return {"status": "PASS", "score": 1.0, "issues": [], "checks": _checks()}

    provider = TinyContextProvider()
    result = VietnameseNaturalnessCritic.review(
        provider, "This source is deliberately long. " * 20, "Bản dịch dài. " * 20, model="mock",
    )

    assert result.status == "ERROR"
    assert result.critic_calls == 0
    assert provider.naturalness_calls == 0
