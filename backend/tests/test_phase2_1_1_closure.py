import json

from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.reference_validator import ReferenceValidator
from app.services.qa.issue_policy import evaluate_issue_resolution
from app.services.qa.result_validator import qa_error
from app.models.canonical import DocumentNode, NodeStatus, NodeType
from app.services.translation.context_assembler import ContextAssembler
from app.services.translation.context_assembler import ContextBudgetExceeded
from app.services.translation.context_memory import ChapterMemory, ChapterMemoryBuilder
from app.services.translation.document_profiler import DocumentProfiler
from app.services.translation.model_capabilities import ModelCapabilities
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.adaptive_chunker import AdaptiveSemanticChunker
from tests.translation_eval import run, run_chapter_benchmark
from app.db.engine import get_project_db
from app.db.models import ChapterModel, NodeModel, ProjectModel, TranslationMemoryModel
from app.services.translation.mock_provider import MockProvider
from app.services.translation.worker import TranslationWorker
from app.services.translation.translation_memory import TranslationMemoryService


def _issue(result, code: str):
    return next((issue for issue in result.issues if issue["code"] == code), None)


def test_reference_validator_accepts_translated_semantic_labels():
    pairs = [
        ("See Figure 3.", "Xem Hình 3."),
        ("See Table 2.", "Xem Bảng 2."),
        ("Read Chapter 4.", "Đọc Chương 4."),
        ("See Section 5.1.", "Xem Mục 5.1."),
        ("See Figure 3 and Table 2.", "Xem Hình 3 và Bảng 2."),
    ]
    for source, target in pairs:
        assert ReferenceValidator.validate(source, target).passed
        assert _issue(TranslationQualityGate().validate(source, target, {}), "REFERENCE_MISMATCH") is None
    assert [token.semantic_key for token in ReferenceValidator.extract("Fig. A.1, Ch. B-2 and Sec. 3.4.2")] == [
        "FIGURE:A.1", "CHAPTER:B-2", "SECTION:3.4.2",
    ]


def test_reference_validator_rejects_wrong_identifier():
    for source, target in [
        ("See Figure 3.", "Xem Hình 4."),
        ("See Table 2.", "Xem Bảng 3."),
    ]:
        result = TranslationQualityGate().validate(source, target, {})
        assert not result.passed
        assert _issue(result, "REFERENCE_MISMATCH")["severity"] == "ERROR"


def test_explicit_negation_loss_is_hard_failure():
    failures = [
        ("The company did not approve the transaction.", "Công ty đã phê duyệt giao dịch."),
        ("Users must not delete this file.", "Người dùng phải xóa tệp này."),
    ]
    for source, target in failures:
        result = TranslationQualityGate().validate(source, target, {})
        issue = _issue(result, "NEGATION_LOSS")
        assert issue and issue["severity"] == "ERROR"
        assert not result.passed
        assert result.hard_fail


def test_preserved_explicit_negation_passes_polarity_gate():
    result = TranslationQualityGate().validate(
        "The company did not approve the transaction.",
        "Công ty không phê duyệt giao dịch.",
        {},
    )
    assert _issue(result, "NEGATION_LOSS") is None


def test_source_budget_hard_invariant_is_kept_in_closure_suite():
    node = DocumentNode(
        id="budget", type=NodeType.PARAGRAPH, content="source " * 450,
        status=NodeStatus.PENDING, order_index=1,
    )
    context = ContextAssembler.assemble_context(
        [node], DocumentProfiler.fallback_profile("GENERAL", node.content), ChapterMemory(), [], {},
        ModelCapabilities(4096, 4096, 120, True, True), [],
        output_contract=PromptBuilder.BATCH_OUTPUT_CONTRACT,
    )
    assert context.token_budget.fits_context_window
    assert not context.token_budget.fits_source_budget
    assert not context.fits
    try:
        context.assert_within_budget()
    except ContextBudgetExceeded:
        pass
    else:
        raise AssertionError("Source vượt hard budget phải bị từ chối")


class NegationDroppingProvider(MockProvider):
    def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.3):
        return [{"node_id": block["id"], "text": "Công ty đã phê duyệt giao dịch."} for block in blocks]

    def translate_single(
        self, text, system_prompt, glossary_terms=None, model=None, temperature=0.3, user_prompt=None,
    ):
        return "Công ty đã phê duyệt giao dịch."


def test_negation_loss_cannot_enter_save_or_tm_path():
    project_id = "phase2_1_1_negation_tm"
    db = get_project_db(project_id)
    try:
        db.query(TranslationMemoryModel).delete()
        db.query(NodeModel).filter(NodeModel.project_id == project_id).delete()
        db.query(ChapterModel).filter(ChapterModel.project_id == project_id).delete()
        db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
        db.commit()
        db.add(ProjectModel(id=project_id, title="Negation", selected_model="mock"))
        db.add(ChapterModel(id="negation_chapter", project_id=project_id, title="Chapter", order_index=0))
        db.add(NodeModel(
            id="negation_node", project_id=project_id, chapter_id="negation_chapter",
            node_type="paragraph", content="The company did not approve the transaction.",
            status="PENDING", order_index=0,
        ))
        db.commit()
        TranslationWorker().translate_project_sync(
            project_id=project_id, model_name="mock", provider=NegationDroppingProvider(),
        )
        db.expire_all()
        node = db.query(NodeModel).filter(NodeModel.id == "negation_node").first()
        assert node.status == "NEEDS_REVIEW"
        assert node.translated_content is None
        assert db.query(TranslationMemoryModel).count() == 0
        try:
            TranslationMemoryService.store(
                db, "The company did not approve the transaction.",
                "Công ty đã phê duyệt giao dịch.",
                style_hash="style", glossary_hash="glossary", prompt_version="test",
            )
        except ValueError as exc:
            assert "NEGATION_LOSS" in str(exc)
        else:
            raise AssertionError("TM service phải tự từ chối candidate mất phủ định")
        assert db.query(TranslationMemoryModel).count() == 0
    finally:
        db.close()


def test_semantic_issue_requires_successful_semantic_re_review():
    failed = evaluate_issue_resolution(
        ["SEMANTIC_MISMATCH"], [],
        {"status": "FAIL", "is_passed": False, "score": 0.2, "issues": ["Sai nghĩa"]},
    )
    assert failed.semantic_review_required
    assert failed.statuses["SEMANTIC_MISMATCH"] == "OPEN"

    passed = evaluate_issue_resolution(
        ["SEMANTIC_MISMATCH"], [],
        {"status": "PASS", "is_passed": True, "score": 1.0, "issues": []},
    )
    assert passed.statuses["SEMANTIC_MISMATCH"] == "RESOLVED"


def test_semantic_reviewer_error_leaves_issue_open_with_qa_error():
    resolution = evaluate_issue_resolution(
        ["AI_SEMANTIC_ERROR"], [], qa_error("provider timeout")
    )
    assert resolution.statuses["AI_SEMANTIC_ERROR"] == "OPEN"
    assert resolution.qa_error == "provider timeout"


def test_multiple_issues_are_resolved_individually():
    resolution = evaluate_issue_resolution(
        ["NUMBER_MISMATCH", "AI_SEMANTIC_ERROR"], [],
        {"status": "FAIL", "is_passed": False, "score": 0.3, "issues": ["Còn sai nghĩa"]},
    )
    assert resolution.statuses == {
        "NUMBER_MISMATCH": "RESOLVED",
        "AI_SEMANTIC_ERROR": "OPEN",
    }


def test_translation_config_remains_target_style_authority():
    project = type("Project", (), {
        "document_type": "GENERAL",
        "translation_mode": "NATURAL",
        "selected_model": "mock",
        "style_guide": {"register": "CONVERSATIONAL", "sentence_style": "FREE"},
    })()
    config = TranslationConfig.from_project(project)
    node = DocumentNode(
        id="authority", type=NodeType.PARAGRAPH, content="A formal source sentence.",
        status=NodeStatus.PENDING, order_index=1,
    )
    profile = DocumentProfiler.fallback_profile("GENERAL", node.content)
    profile.register = "formal"
    context = ContextAssembler.assemble_context(
        [node], profile, ChapterMemory(), [], {},
        ModelCapabilities(4096, 4096, 1200, True, True), [],
        system_prompt=PromptBuilder.build_system_prompt(config),
        output_contract=PromptBuilder.BATCH_OUTPUT_CONTRACT,
    )
    contract = context.system_prompt + "\n" + PromptBuilder.build_batch_prompt([node], context)
    assert '"source_register":"formal"' in contract
    assert "Register: CONVERSATIONAL" in contract
    assert "Sentence restructuring: FREE" in contract
    assert "must not override TARGET register" in contract


class StructuredMemoryProvider:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def build_chapter_memory(self, text_sample, chapter_title, document_type, model=None):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _memory_response(summary="Tóm tắt có cấu trúc."):
    return {
        "summary": summary,
        "entities": [{"source": "Acme", "preferred": "Acme"}],
        "key_concepts": ["growth"],
        "tone": "analytical",
        "pronoun_notes": [],
        "terminology": [{"source": "cash flow", "preferred": "dòng tiền"}],
        "important_facts": ["Revenue increased."],
        "style_notes": [],
    }


def test_valid_structured_chapter_memory_populates_and_uses_cache(tmp_path):
    node = DocumentNode(
        id="memory", type=NodeType.PARAGRAPH, content="Acme improved cash flow.",
        status=NodeStatus.PENDING, order_index=1,
    )
    provider = StructuredMemoryProvider(_memory_response())
    first = ChapterMemoryBuilder.load_or_create(
        "chapter", "Growth", [node], tmp_path, {"cash flow": "dòng tiền"},
        provider, "mock", "BUSINESS",
    )
    second = ChapterMemoryBuilder.load_or_create(
        "chapter", "Growth", [node], tmp_path, {"cash flow": "dòng tiền"},
        provider, "mock", "BUSINESS",
    )
    assert first.summary == "Tóm tắt có cấu trúc."
    assert first.entities == [{"source": "Acme", "preferred": "Acme"}]
    assert second.to_dict() == first.to_dict()
    assert provider.calls == 1


def test_malformed_chapter_memory_falls_back_and_cache_invalidates(tmp_path):
    node = DocumentNode(
        id="memory", type=NodeType.PARAGRAPH, content="Acme improved cash flow.",
        status=NodeStatus.PENDING, order_index=1,
    )
    provider = StructuredMemoryProvider({"summary": []})
    fallback = ChapterMemoryBuilder.load_or_create(
        "chapter", "Growth", [node], tmp_path, {}, provider, "mock", "BUSINESS",
    )
    assert fallback.summary
    assert fallback.entities
    assert provider.calls == 1

    changed_source = node.model_copy(update={"content": "Beta changed the operating model."})
    provider.response = _memory_response("Nguồn đã thay đổi.")
    ChapterMemoryBuilder.load_or_create(
        "chapter", "Growth", [changed_source], tmp_path, {}, provider, "mock", "BUSINESS",
    )
    assert provider.calls == 2
    ChapterMemoryBuilder.load_or_create(
        "chapter", "Growth", [changed_source], tmp_path, {"model": "mô hình"},
        provider, "mock", "BUSINESS",
    )
    assert provider.calls == 3


def test_benchmark_isolates_domains_and_independent_nodes(tmp_path):
    dataset = tmp_path / "nodes.jsonl"
    dataset.write_text(
        '\n'.join([
            '{"domain":"GENERAL","source":"The plan remains active."}',
            '{"domain":"FINANCE","source":"The yield was 2%."}',
        ]),
        encoding="utf-8",
    )
    human_csv = tmp_path / "human.csv"
    report = run(dataset, human_csv=human_csv)
    assert set(report["domains"]) == {"GENERAL", "FINANCE"}
    assert len({group["project_id"] for group in report["domains"].values()}) == 2
    assert all(group["metadata"]["domain"] == domain for domain, group in report["domains"].items())
    assert "phase2_1_1_translation" in human_csv.read_text(encoding="utf-8-sig").splitlines()[0]


def test_chapter_benchmark_uses_one_project_per_book(tmp_path):
    dataset = tmp_path / "chapters.json"
    dataset.write_text(json.dumps([
        {
            "book_id": "business-one", "domain": "BUSINESS", "chapter_title": "Business",
            "node_count": 11, "entities": ["Acme"], "locked_glossary": {},
            "source_templates": ["Acme reviewed plan {n}."],
        },
        {
            "book_id": "technical-one", "domain": "TECHNICAL", "chapter_title": "Technical",
            "node_count": 11, "entities": ["NodeManager"], "locked_glossary": {},
            "source_templates": ["NodeManager handled request {n}."],
        },
    ]), encoding="utf-8")
    report = run_chapter_benchmark(dataset)
    assert {book["domain"] for book in report["books"].values()} == {"BUSINESS", "TECHNICAL"}
    assert len({book["project_id"] for book in report["books"].values()}) == 2


def test_table_and_footnote_are_isolated_chunks():
    nodes = [
        DocumentNode(id="p1", type=NodeType.PARAGRAPH, content="Paragraph one.", status=NodeStatus.PENDING, order_index=1),
        DocumentNode(id="t1", type=NodeType.TABLE, content="A | B", status=NodeStatus.PENDING, order_index=2),
        DocumentNode(id="p2", type=NodeType.PARAGRAPH, content="Paragraph two.", status=NodeStatus.PENDING, order_index=3),
        DocumentNode(id="f1", type=NodeType.FOOTNOTE, content="Footnote.", status=NodeStatus.PENDING, order_index=4),
    ]
    chunks = AdaptiveSemanticChunker.chunk_nodes(nodes, source_budget=1200)
    assert [[node.id for node in chunk.nodes] for chunk in chunks] == [["p1"], ["t1"], ["p2"], ["f1"]]
