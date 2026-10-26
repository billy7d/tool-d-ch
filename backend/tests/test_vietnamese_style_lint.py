import json
from pathlib import Path

from app.services.qa.vietnamese_naturalness_critic import VietnameseNaturalnessCritic, VietnameseNaturalnessResult
from app.services.qa.vietnamese_style_lint import VietnameseStyleLint
from app.services.translation.few_shot_library import CURATED_FEW_SHOTS
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.vietnamese_editorial_assurance import TranslationPublicationAssuranceService


EVAL_PATH = Path(__file__).parent / "fixtures" / "translation_eval" / "vietnamese_naturalness_eval.jsonl"


def _eval_rows():
    return [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rules(text, source="", document_type="GENERAL"):
    return {item.rule for item in VietnameseStyleLint.lint(text, source, document_type)}


def test_lint_has_no_false_positive_on_natural_benchmark_candidates():
    flagged = [
        row["id"] for row in _eval_rows()
        if VietnameseStyleLint.lint(row["natural_candidate"], row["source"], row["domain"])
    ]
    assert flagged == []


def test_lint_catches_majority_of_literal_benchmark_candidates():
    rows = _eval_rows()
    caught = sum(bool(VietnameseStyleLint.lint(row["literal_candidate"], row["source"], row["domain"])) for row in rows)
    assert caught / len(rows) >= 0.6


def test_curated_few_shot_targets_are_lint_clean():
    # Few-shot dạy model mạnh hơn chỉ dẫn, nên ví dụ không được chứa chính mẫu câu mà prompt cấm.
    dirty = {
        example.example_id: [item.rule for item in VietnameseStyleLint.lint(example.target, example.source, domain)]
        for domain, examples in CURATED_FEW_SHOTS.items()
        for example in examples
        if VietnameseStyleLint.lint(example.target, example.source, domain)
    }
    assert dirty == {}


def test_core_calque_rules():
    assert "PASSIVE_BY_AGENT" in _rules("Đề xuất đã được phê duyệt bởi hội đồng.")
    assert "MOT_CACH_ADVERB" in _rules("Anh ấy giải thích một cách rõ ràng.")
    assert "LIGHT_VERB_VIEC" in _rules("Nhóm sẽ tiến hành việc kiểm tra.")
    assert "IT_IS_ADJ_TO" in _rules("Các hướng dẫn là dễ dàng để làm theo.")
    assert "CALQUE_PHRASE" in _rules("Bản cập nhật đã dẫn đến trong một sự cải thiện.")
    assert "DUPLICATED_FUNCTION_WORD" in _rules("Đây là kết quả của của nhóm.")


def test_noun_phrase_mot_cach_is_not_flagged():
    assert "MOT_CACH_ADVERB" not in _rules("Đây là một cách tiếp cận mới.")
    assert "MOT_CACH_ADVERB" not in _rules("Có một cách đơn giản để giải quyết vấn đề.")


def test_no_for_person_requires_person_pronoun_in_source():
    assert "NO_FOR_PERSON" in _rules("Nó mở cửa bước vào.", "He opened the door and walked in.")
    assert "NO_FOR_PERSON" not in _rules("Nó chạy rất nhanh.", "It runs very fast.")


def test_formal_domains_downgrade_passive_to_info():
    text = "Hợp đồng được ký bởi hai bên."
    general = VietnameseStyleLint.lint(text, document_type="GENERAL")
    legal = VietnameseStyleLint.lint(text, document_type="LEGAL")
    assert [item.severity for item in general if item.rule == "PASSIVE_BY_AGENT"] == ["WARNING"]
    assert [item.severity for item in legal if item.rule == "PASSIVE_BY_AGENT"] == ["INFO"]


def test_critic_forwards_lint_hints_and_keeps_them_on_result():
    captured = {}

    class Provider:
        def effective_context_window(self, model):
            return 8192

        def review_naturalness(self, **kwargs):
            captured.update(kwargs)
            checks = {key: "PASS" for key in (
                "literal_calque", "word_order", "collocation", "cohesion", "pronoun_reference",
                "register", "redundancy", "nominalization", "passive_voice", "sentence_flow",
            )}
            return {"status": "PASS", "score": 0.9, "checks": checks, "issues": []}

    result = VietnameseNaturalnessCritic.review(
        Provider(), "The proposal was approved by the board.", "Đề xuất đã được phê duyệt bởi hội đồng.",
    )
    assert [hint["rule"] for hint in captured["style_hints"]] == ["PASSIVE_BY_AGENT"]
    assert [hint["rule"] for hint in result.lint_issues] == ["PASSIVE_BY_AGENT"]
    # Lint chỉ là gợi ý: critic PASS thì kết quả vẫn PASS.
    assert result.passed()


def _result_with(checks_failed, lint_rules):
    checks = {key: "PASS" for key in (
        "literal_calque", "word_order", "collocation", "cohesion", "pronoun_reference",
        "register", "redundancy", "nominalization", "passive_voice", "sentence_flow",
    )}
    for key in checks_failed:
        checks[key] = "FAIL"
    text = "Đề xuất đã được phê duyệt bởi hội đồng và anh giải thích một cách rõ ràng."
    lint = [item.to_hint() for item in VietnameseStyleLint.lint(text) if item.rule in lint_rules]
    return VietnameseNaturalnessResult("FAIL", 0.7, [], checks, lint_issues=lint)


def test_rewrite_receives_lint_spans_only_for_checks_the_critic_failed():
    result = _result_with(["passive_voice"], {"PASSIVE_BY_AGENT", "MOT_CACH_ADVERB"})
    issues = TranslationPublicationAssuranceService._rewrite_issues(result)
    assert [item["type"] for item in issues] == ["PASSIVE_VOICE"]
    assert issues[0]["message"].startswith("[lint:PASSIVE_BY_AGENT]")


def test_ollama_single_translation_uses_low_repeat_penalty():
    provider = OllamaProvider()
    sent = {}

    class Response:
        status_code = 200

        def json(self):
            return {"message": {"content": "Xin chào."}}

    def post(url, json=None, timeout=None):
        sent.update(json)
        return Response()

    provider.session.post = post
    provider.translate_single("Hello.", "system", model="qwen2.5:7b")
    assert sent["options"]["repeat_penalty"] == OllamaProvider.VIETNAMESE_REPEAT_PENALTY
    assert OllamaProvider.VIETNAMESE_REPEAT_PENALTY <= 1.1
