import json

from app.db.engine import get_project_db
from app.db.models import ChapterModel, NodeModel, ProjectModel, TranslationMemoryModel
from app.services.qa.result_validator import validate_qa_result
from app.services.translation.glossary_validator import GlossaryValidator
from app.services.translation.json_parser import parse_translation_batch_strict
from app.services.translation.language_validator import validate_target_language
from app.services.translation.mock_provider import MockProvider
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.openai_local_provider import OpenAILocalProvider
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.translation_signature import build_translation_signature
from app.services.translation.worker import TranslationWorker


def test_locked_glossary_pass_and_fail():
    glossary = {"cash flow": "dòng tiền"}
    passed = GlossaryValidator.validate(
        "The company improved its cash flow.",
        "Công ty đã cải thiện dòng tiền.",
        glossary,
    )
    failed = GlossaryValidator.validate(
        "The company improved its cash flow.",
        "Công ty đã cải thiện luồng tiền.",
        glossary,
    )
    assert passed.passed is True
    assert failed.passed is False
    assert failed.violations[0].reason == "TARGET_TERM_MISSING"


def test_glossary_word_boundary_avoids_false_substring():
    result = GlossaryValidator.validate(
        "Capitalization rules are important.",
        "Quy tắc viết hoa rất quan trọng.",
        {"capital": "vốn"},
    )
    assert result.passed is True


def test_language_validation_rejects_cjk_and_english_but_allows_names_and_acronyms():
    cjk = validate_target_language("AI technology is growing.", "Công nghệ 人工智能 đang phát triển.")
    english = validate_target_language("The company increased revenue.", "The company increased revenue.")
    proper_name = validate_target_language("OpenAI released a new version.", "OpenAI phát hành phiên bản mới.")
    acronym = validate_target_language("This API uses HTTP.", "API này sử dụng HTTP.")
    assert cjk.reason == "FOREIGN_SCRIPT_CONTAMINATION"
    assert english.reason == "WRONG_TARGET_LANGUAGE"
    assert proper_name.passed is True
    assert acronym.passed is True


def test_quality_gate_preserves_content_instead_of_stripping_cjk():
    candidate = "Công nghệ 人工智能 đang phát triển."
    result = TranslationQualityGate().validate("AI technology is growing.", candidate, {})
    assert result.passed is False
    assert any(issue["code"] == "FOREIGN_SCRIPT_CONTAMINATION" for issue in result.issues)
    assert "人工智能" in candidate


def test_quality_gate_numbers_urls_references_and_glossary():
    result = TranslationQualityGate().validate(
        "See Figure 1: cash flow rose 20% at https://example.com in 2026.",
        "Xem Hình: luồng tiền tăng tại trang web trong năm 2026.",
        {"cash flow": "dòng tiền"},
    )
    codes = {issue["code"] for issue in result.issues}
    assert {"NUMBER_MISMATCH", "URL_MISMATCH", "REFERENCE_MISMATCH", "GLOSSARY_MISMATCH"} <= codes


def test_strict_batch_parser_reports_mapping_errors_without_position_fallback():
    missing = parse_translation_batch_strict(
        json.dumps({"translations": [{"node_id": "A", "text": "Một"}, {"node_id": "C", "text": "Ba"}]}),
        ["A", "B", "C"],
    )
    duplicate = parse_translation_batch_strict(
        json.dumps({"translations": [
            {"node_id": "A", "text": "Một"},
            {"node_id": "B", "text": "Hai"},
            {"node_id": "B", "text": "Hai nữa"},
            {"node_id": "C", "text": "Ba"},
        ]}),
        ["A", "B", "C"],
    )
    unknown = parse_translation_batch_strict(
        json.dumps({"translations": [
            {"node_id": "A", "text": "Một"},
            {"node_id": "B", "text": "Hai"},
            {"node_id": "C", "text": "Ba"},
            {"node_id": "D", "text": "Bốn"},
        ]}),
        ["A", "B", "C"],
    )
    no_ids = parse_translation_batch_strict(
        json.dumps({"translations": [{"text": "Một"}, {"text": "Hai"}]}),
        ["A", "B"],
    )
    assert missing.missing_ids == ["B"]
    assert duplicate.duplicate_ids == ["B"]
    assert unknown.unknown_ids == ["D"]
    assert no_ids.missing_ids == ["A", "B"]
    assert no_ids.translations == []


def test_translation_signature_isolates_mode_and_glossary():
    base = dict(
        source_language="en",
        target_language="vi",
        document_type="GENERAL",
        style_guide={},
    )
    natural = build_translation_signature(
        **base, translation_mode="NATURAL", locked_glossary={"leverage": "đòn bẩy"}
    )
    technical = build_translation_signature(
        **base, translation_mode="TECHNICAL", locked_glossary={"leverage": "đòn bẩy"}
    )
    changed_glossary = build_translation_signature(
        **base, translation_mode="NATURAL", locked_glossary={"leverage": "hệ số đòn bẩy"}
    )
    assert natural.style_hash != technical.style_hash
    assert natural.glossary_hash != changed_glossary.glossary_hash
    assert natural.prompt_version == "translation-v3.3-naturalness-p0"


def test_qa_schema_is_fail_closed():
    malformed = validate_qa_result({"score": 1.5, "issues": []})
    assert malformed["status"] == "ERROR"
    assert malformed["is_passed"] is False
    assert malformed["score"] is None


def test_qa_provider_timeout_is_fail_closed(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise TimeoutError("provider timeout")

    for provider in (OllamaProvider(), OpenAILocalProvider()):
        monkeypatch.setattr(provider.session, "post", raise_timeout)
        result = provider.review_translation("Source", "Bản dịch", {}, model="mock")
        assert result["status"] == "ERROR"
        assert result["is_passed"] is False
        assert result["score"] is None


class AlwaysEnglishProvider(MockProvider):
    def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.3):
        return [{"node_id": block["id"], "text": "The company increased revenue."} for block in blocks]

    def translate_single(self, text, system_prompt, glossary_terms=None, model=None, temperature=0.3):
        return "The company increased revenue."


def test_invalid_translation_never_enters_tm():
    project_id = "test_phase1_invalid_tm"
    db = get_project_db(project_id)
    try:
        db.query(TranslationMemoryModel).delete()
        db.query(NodeModel).filter(NodeModel.project_id == project_id).delete()
        db.query(ChapterModel).filter(ChapterModel.project_id == project_id).delete()
        db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
        db.commit()
        db.add(ProjectModel(id=project_id, title="Phase 1", selected_model="mock"))
        db.add(ChapterModel(id="phase1_ch", project_id=project_id, title="Chương", order_index=0))
        db.add(NodeModel(
            id="phase1_node",
            project_id=project_id,
            chapter_id="phase1_ch",
            node_type="paragraph",
            content="The company increased revenue.",
            order_index=0,
            status="PENDING",
            version=1,
        ))
        db.commit()

        TranslationWorker().translate_project_sync(
            project_id=project_id,
            model_name="mock",
            provider=AlwaysEnglishProvider(),
        )
        db.expire_all()
        node = db.query(NodeModel).filter(NodeModel.id == "phase1_node").first()
        assert node.status == "NEEDS_REVIEW"
        assert node.translated_content is None
        assert db.query(TranslationMemoryModel).count() == 0
    finally:
        db.close()
