from app.db.engine import get_project_db
from app.db.models import NodeModel, ProjectModel
from app.db.repository import ProjectRepository
from app.models.canonical import DocumentNode, NodeStatus, NodeType
from app.services.translation.context_assembler import ContextAssembler, ContextBudgetExceeded
from app.services.translation.context_memory import BilingualContextItem, ChapterMemory
from app.services.translation.document_profiler import DocumentProfiler
from app.services.translation.language_validator import validate_target_language
from app.services.translation.model_capabilities import ModelCapabilities
from app.services.translation.numeric_validator import NumericValidator
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.translation_config import TranslationConfig
from app.services.translation.translation_signature import build_translation_signature_from_config


def _node(index: int, text: str) -> DocumentNode:
    return DocumentNode(
        id=f"n{index}", type=NodeType.PARAGRAPH, content=text,
        status=NodeStatus.PENDING, order_index=index,
    )


def test_preview_and_production_share_normalized_config():
    project = ProjectModel(
        id="config", title="Config", document_type="LEGAL", translation_mode="NATURAL",
        selected_model="mock", style_guide={"register": "formal", "sentence_style": "preserve"},
    )
    production = TranslationConfig.from_project(project)
    preview = TranslationConfig.from_project(
        project,
        model_override="mock",
        document_type_override="LEGAL",
        translation_mode_override="NATURAL",
        register_override="FORMAL",
        sentence_style_override="PRESERVE",
    )
    assert production == preview
    assert production.register == "FORMAL"
    assert production.sentence_style == "PRESERVE"


def test_prompt_contracts_do_not_contradict_each_other():
    assert "JSON" in PromptBuilder.BATCH_OUTPUT_CONTRACT
    assert "plain" not in PromptBuilder.BATCH_OUTPUT_CONTRACT.lower()
    assert "plain text" in PromptBuilder.SINGLE_OUTPUT_CONTRACT.lower()
    assert "JSON" not in PromptBuilder.SINGLE_OUTPUT_CONTRACT


def test_hard_budget_trims_optional_context_and_preserves_source():
    node = _node(1, "The Federal Reserve set the rate to 1.20%.")
    profile = DocumentProfiler.fallback_profile("FINANCE", node.content)
    profile.summary = "document " * 5000
    memory = ChapterMemory(summary="chapter " * 5000)
    rolling = [
        BilingualContextItem(node_id=f"r{i}", source="source " * 200, translation="bản dịch " * 200)
        for i in range(8)
    ]
    context = ContextAssembler.assemble_context(
        [node], profile, memory, rolling,
        {f"term{i}": f"thuật ngữ {i}" for i in range(500)} | {"Federal Reserve": "Cục Dự trữ Liên bang Mỹ"},
        ModelCapabilities(4096, 4096, 1200, True, True),
        [{"source": "example " * 1000, "target": "ví dụ " * 1000}],
        system_prompt="system " * 200,
        output_contract=PromptBuilder.BATCH_OUTPUT_CONTRACT,
    )
    assert context.fits
    assert context.token_budget.total_estimated_tokens <= 4096
    assert context.token_budget.source_tokens > 0
    assert context.glossary == {"Federal Reserve": "Cục Dự trữ Liên bang Mỹ"}
    assert context.trim_steps


def test_source_budget_is_hard_even_when_request_fits_context_window():
    node = _node(1, "source " * 450)
    profile = DocumentProfiler.fallback_profile("GENERAL", node.content)
    context = ContextAssembler.assemble_context(
        [node], profile, ChapterMemory(), [], {},
        ModelCapabilities(4096, 4096, 120, True, True), [],
        system_prompt="Dịch chính xác sang tiếng Việt.",
        output_contract=PromptBuilder.BATCH_OUTPUT_CONTRACT,
    )

    assert context.token_budget.fits_context_window
    assert not context.token_budget.fits_source_budget
    assert not context.token_budget.fits
    assert context.token_budget.source_tokens > context.token_budget.available_source_tokens

    try:
        context.assert_within_budget()
    except ContextBudgetExceeded as exc:
        assert "vượt ngân sách nguồn" in str(exc)
    else:
        raise AssertionError("Source vượt ngân sách phải bị từ chối")


def test_full_glossary_validation_remains_after_inference_filtering():
    glossary = {f"term{i}": f"thuật ngữ {i}" for i in range(500)}
    glossary.update({"alpha": "an-pha", "beta": "bê-ta"})
    node = _node(1, "Alpha and beta are related.")
    profile = DocumentProfiler.fallback_profile("GENERAL", node.content)
    context = ContextAssembler.assemble_context(
        [node], profile, ChapterMemory(), [], glossary,
        ModelCapabilities(4096, 4096, 1200, True, True), [],
    )
    assert set(context.glossary) == {"alpha", "beta"}
    result = TranslationQualityGate().validate(node.content, "An-pha có liên quan.", glossary)
    assert any(issue["code"] == "GLOSSARY_MISMATCH" for issue in result.issues)


def test_numeric_language_and_negation_hardening():
    negative = NumericValidator.validate(
        "The temperature can fall to -10°C.",
        "Nhiệt độ có thể giảm xuống 10°C.",
    )
    decimal = NumericValidator.validate("The ratio is 1.20.", "Tỷ lệ là 12,0.")
    leakage = validate_target_language(
        "Quantum entanglement enables nonlocal correlations.",
        "Quantum entanglement enables nonlocal correlations.",
    )
    negation = TranslationQualityGate().validate(
        "The company did not approve the transaction.",
        "Công ty đã phê duyệt giao dịch.",
        {},
    )
    assert not negative.passed
    assert not decimal.passed
    assert leakage.reason == "WRONG_TARGET_LANGUAGE"
    assert any(issue["code"] == "NEGATION_LOSS" and issue["severity"] == "ERROR" for issue in negation.issues)
    assert not negation.passed


def test_non_translatable_progress_and_review_terminal_accounting():
    project_id = "phase2_1_stats"
    db = get_project_db(project_id)
    db.query(NodeModel).filter(NodeModel.project_id == project_id).delete()
    db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
    db.add(ProjectModel(id=project_id, title="Stats"))
    db.commit()
    for index in range(90):
        db.add(NodeModel(
            id=f"p{index}", project_id=project_id, node_type="paragraph",
            content=f"Paragraph {index}", status="TRANSLATED", order_index=index,
        ))
    for index in range(10):
        db.add(NodeModel(
            id=f"i{index}", project_id=project_id, node_type="image",
            content="", status="PENDING", order_index=100 + index,
        ))
    db.commit()
    stats = ProjectRepository(db).get_project_stats(project_id)
    assert stats["total_nodes"] == 100
    assert stats["translatable_nodes"] == 90
    assert stats["translated_nodes"] == 90
    assert stats["skipped_nodes"] == 10
    assert stats["progress_percent"] == 100.0

    db.query(NodeModel).filter(NodeModel.id == "p89").update({"status": "NEEDS_REVIEW"})
    db.commit()
    review_stats = ProjectRepository(db).get_project_stats(project_id)
    assert round(review_stats["progress_percent"], 1) == 98.9
    assert review_stats["terminal_nodes"] == 90
    db.close()


def test_translation_signature_changes_with_normalized_style():
    project = ProjectModel(id="signature", title="Signature", selected_model="mock")
    conversational = TranslationConfig.from_project(project, register_override="CONVERSATIONAL")
    formal = TranslationConfig.from_project(project, register_override="FORMAL")
    glossary = {"risk": "rủi ro"}
    assert build_translation_signature_from_config(conversational, glossary).style_hash != build_translation_signature_from_config(formal, glossary).style_hash
