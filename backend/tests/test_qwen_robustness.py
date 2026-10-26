"""Các lỗi đã gặp khi chạy qwen2.5:7b thật: JSON critic lệch schema, critic cho PASS bản sát chữ,
chèn nội dung từ đoạn trước, rơi dấu ngoặc kép và lọt chữ Hán."""
import json
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Base, ChapterModel, NodeModel, ProjectModel
from app.services.qa.vietnamese_naturalness_critic import (
    NATURALNESS_CHECKS,
    VietnameseNaturalnessCritic,
    normalize_naturalness_payload,
    validate_naturalness_result,
)
from app.services.translation.contextual_engine import ContextualTranslationEngine
from app.services.translation.mock_provider import MockProvider
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.translation_config import TranslationConfig


LITERAL = "Đề xuất đã được phê duyệt bởi hội đồng quản trị."
NATURAL = "Hội đồng quản trị đã phê duyệt đề xuất."
SOURCE = "The proposal was approved by the board."


def _checks(value="PASS"):
    return {key: value for key in NATURALNESS_CHECKS}


# ---------- Chuẩn hóa JSON lệch schema ----------

def test_issues_as_object_and_alias_severity_are_repaired():
    payload, repairs = normalize_naturalness_payload({
        "status": "FAIL", "score": 7,
        "issues": {"type": "passive", "severity": "medium", "description": "Bị động kiểu tiếng Anh."},
    })
    result = validate_naturalness_result(payload)
    assert result.status == "FAIL"
    assert result.score == 0.7
    assert result.issues[0]["type"] == "PASSIVE_VOICE" and result.issues[0]["severity"] == "WARNING"
    assert result.checks["passive_voice"] == "FAIL"
    assert {"issues_not_list", "score_scale", "check_derived"} <= set(repairs)


def test_missing_checks_and_status_are_derived_from_issues():
    payload, _ = normalize_naturalness_payload({"Score": "0.92", "Issues": []})
    result = validate_naturalness_result(payload)
    assert result.status == "PASS" and result.passed()


def test_pass_with_blocking_issue_becomes_fail_not_error():
    payload, repairs = normalize_naturalness_payload({
        "status": "PASS", "score": 1.0, "checks": _checks(),
        "issues": [{"type": "LITERAL_CALQUE", "severity": "high", "message": "Dịch sát."}],
    })
    assert payload["status"] == "FAIL" and "pass_contradiction" in repairs


def test_fail_without_any_reason_is_still_rejected():
    with pytest.raises(ValueError):
        normalize_naturalness_payload({"status": "FAIL", "score": 0.4, "issues": [], "checks": _checks()})


def test_json_embedded_in_text_is_extracted():
    payload, repairs = normalize_naturalness_payload('Kết quả: {"status": "PASS", "score": 0.9, "issues": []} xong')
    assert payload["status"] == "PASS" and "json_extracted" in repairs


# ---------- Lint là tín hiệu chính ----------

class _Critic(MockProvider):
    def __init__(self, response):
        self.response = response

    def review_naturalness(self, **kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_critic_pass_on_literal_text_is_overridden_by_lint():
    # Đúng hành vi đo được: qwen2.5:7b cho PASS 1.0 cả bản sát chữ.
    result = VietnameseNaturalnessCritic.review(
        _Critic({"status": "PASS", "score": 1.0, "issues": [], "checks": _checks()}), SOURCE, LITERAL, model="m",
    )
    assert result.verdict_source == "LINT_OVERRIDE"
    assert result.status == "FAIL" and result.rewrite_allowed()


def test_unusable_json_falls_back_to_lint_verdict():
    broken = json.JSONDecodeError("Expecting value", "not json", 0)
    literal = VietnameseNaturalnessCritic.review(_Critic(broken), SOURCE, LITERAL, model="m")
    natural = VietnameseNaturalnessCritic.review(_Critic(broken), SOURCE, NATURAL, model="m")
    assert literal.verdict_source == natural.verdict_source == "LINT_FALLBACK"
    assert literal.status == "FAIL" and literal.rewrite_allowed()
    assert natural.status == "PASS" and natural.passed()


def test_lint_fallback_can_be_disabled():
    result = VietnameseNaturalnessCritic.review(
        _Critic(ValueError("bad schema")), SOURCE, NATURAL, model="m", lint_fallback=False,
    )
    assert result.status == "ERROR"


def test_infrastructure_errors_stay_fail_closed():
    result = VietnameseNaturalnessCritic.review(_Critic(RuntimeError("Ollama timeout")), SOURCE, NATURAL, model="m")
    assert result.status == "ERROR" and not result.passed()


# ---------- Gate chống chèn nội dung ----------

def _codes(source, target, context=None):
    return {issue["code"] for issue in TranslationQualityGate().validate(source, target, {}, context).issues}


def test_gate_flags_added_question_from_neighbour_paragraph():
    source = 'Bob looked at Alice and said, "I promised you I would return, didn\'t I?"'
    inserted = 'Bob nhìn Alice và nói: "Em đã chờ anh suốt mùa đông, anh có nhớ không? Anh đã hứa sẽ trở lại, đúng không?"'
    assert "CONTENT_ADDITION" in _codes(source, inserted)
    assert "CONTENT_ADDITION" not in _codes(source, 'Bob nhìn Alice và nói: "Anh đã hứa sẽ trở lại, đúng không?"')


def test_gate_flags_dropped_opening_quote():
    source = '"You came back," Alice said to Bob.'
    assert "QUOTE_UNBALANCED" in _codes(source, 'Anh về rồi," Alice nói với Bob.')
    assert "QUOTE_UNBALANCED" not in _codes(source, '“Anh về rồi”, Alice nói với Bob.')


def test_gate_flags_sentence_copied_from_previous_translation():
    previous = ["Em đã chờ anh suốt cả mùa đông dài lạnh giá."]
    copied = "Anh nhìn cô. Em đã chờ anh suốt cả mùa đông dài lạnh giá."
    assert "CONTEXT_LEAK" in _codes("He looked at her.", copied, previous)
    assert "CONTEXT_LEAK" not in _codes("He looked at her.", "Anh nhìn cô.", previous)


# ---------- Khôi phục khi lọt chữ Hán ----------

class _ChineseProvider(MockProvider):
    """Prompt dài luôn ra tiếng Trung (như câu pháp lý khi test); prompt tiếng Việt ngắn thì tùy kịch bản."""

    def __init__(self, minimal_answer):
        self.minimal_answer = minimal_answer
        self.calls = []

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.15, user_prompt=None):
        self.calls.append(system_prompt[:30])
        if system_prompt == ContextualTranslationEngine.FOREIGN_SCRIPT_SYSTEM_PROMPT:
            if "Dịch cụm tiếng Trung" in (user_prompt or ""):
                return "không được tiết lộ thông tin mật"
            return self.minimal_answer
        return "Nhà cung cấp不得披露机密信息，除非法律要求。"


@pytest.fixture()
def engine_factory():
    original = settings.PROJECTS_DIR
    temp_dir = tempfile.TemporaryDirectory(prefix="qwen-robustness-")
    settings.PROJECTS_DIR = Path(temp_dir.name)
    db_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(db_engine)
    db = sessionmaker(bind=db_engine)()

    def build(provider):
        project = ProjectModel(id="cjk", title="CJK", selected_model="mock", document_type="LEGAL", translation_mode="NATURAL")
        chapter = ChapterModel(id="cjk_ch", project_id="cjk", title="Điều khoản", order_index=0)
        node = NodeModel(id="cjk_node", project_id="cjk", chapter_id="cjk_ch", node_type="paragraph",
                         content="The supplier must not disclose Confidential Information except as required by law.",
                         status="PENDING", order_index=0)
        db.add_all([project, chapter, node])
        db.commit()
        config = TranslationConfig.from_project(project, model_override="mock")
        engine = ContextualTranslationEngine(db, project, provider, config, {})
        return engine, chapter, engine.canonical_node(node)

    yield build
    db.close()
    db_engine.dispose()
    settings.PROJECTS_DIR = original
    temp_dir.cleanup()


def test_chinese_output_is_recovered_with_minimal_vietnamese_prompt(engine_factory):
    clean = "Nhà cung cấp không được tiết lộ Thông tin Mật, trừ khi pháp luật yêu cầu."
    engine, chapter, node = engine_factory(_ChineseProvider(clean))
    result = engine.translate_node(chapter, node)
    assert result.passed and result.translated_text == clean


def test_leftover_chinese_fragments_are_translated_in_place(engine_factory):
    engine, chapter, node = engine_factory(_ChineseProvider("Nhà cung cấp 不得披露机密信息, trừ khi pháp luật yêu cầu."))
    result = engine.translate_node(chapter, node)
    assert result.passed
    assert "không được tiết lộ thông tin mật" in result.translated_text
    assert not any("一" <= char <= "鿿" for char in result.translated_text)


def test_gate_flags_dropped_quote_pair_even_when_count_is_even():
    source = '"I will stay," Bob told Alice. "You are the reason I came home."'
    assert "QUOTE_UNBALANCED" in _codes(source, 'Em sẽ ở lại," Bob nói với Alice. "Anh là lý do em trở về nhà.')
    assert "QUOTE_UNBALANCED" not in _codes(source, '“Anh sẽ ở lại”, Bob nói với Alice. “Em là lý do anh trở về nhà.”')
    assert "QUOTE_UNBALANCED" not in _codes(source, '– Anh sẽ ở lại, – Bob nói với Alice.')


def test_provider_keeps_dialogue_quotes_but_unwraps_whole_output_wrapper():
    from app.services.translation.provider_base import unwrap_quoted_output

    dialogue = '"Anh về rồi," cô nói. "Em đã chờ anh."'
    assert unwrap_quoted_output(dialogue, '"You came back," she said. "I waited."') == dialogue
    assert unwrap_quoted_output('"Doanh thu tăng 12%."', "Revenue rose 12%.") == "Doanh thu tăng 12%."
    # Nguồn là một câu thoại trọn vẹn thì cặp ngoặc là của câu thoại, phải giữ.
    assert unwrap_quoted_output('"Đợi anh nhé."', '"Wait for me."') == '"Đợi anh nhé."'
