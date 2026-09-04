from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, GlossaryModel
from app.services.translation.context_assembler import ContextAssembler, TranslationContext
from app.services.translation.document_profiler import DocumentProfiler
from app.services.translation.context_memory import ChapterMemory
from app.services.translation.glossary_service import GlossaryService
from app.services.translation.model_capabilities import ModelCapabilities
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.quality_gate import TranslationQualityGate


def test_hard_mismatch_fails_and_allowed_variant_passes():
    hard = [{
        "source_term": "OpenAI", "preferred_target": "OpenAI",
        "allowed_variants": [], "lock_level": "HARD",
    }]
    assert not TranslationQualityGate().validate("OpenAI published a report.", "Công ty xuất bản báo cáo.", hard).passed
    assert TranslationQualityGate().validate("OpenAI published a report.", "OpenAI đã xuất bản báo cáo.", hard).passed

    soft = [{
        "source_term": "margin", "preferred_target": "biên lợi nhuận",
        "allowed_variants": ["biên"], "sense_hint": "finance/profitability",
        "domain": "FINANCE", "lock_level": "SOFT",
    }]
    result = TranslationQualityGate().validate("The margin increased.", "Biên tăng.", soft)
    assert result.passed
    assert not result.issues


def test_soft_mismatch_is_warning_and_prompt_does_not_force_regex_replacement():
    soft = [{
        "source_term": "interest", "preferred_target": "tiền lãi",
        "allowed_variants": ["lãi suất"], "sense_hint": "finance/loan",
        "domain": "FINANCE", "lock_level": "SOFT",
    }]
    result = TranslationQualityGate().validate(
        "Interest on the loan increased.", "Chi phí vay tăng.", soft,
    )
    assert result.passed
    assert any(issue["code"] == "GLOSSARY_PREFERENCE" and issue["severity"] == "WARNING" for issue in result.issues)

    context = TranslationContext(
        soft_glossary={"interest": "tiền lãi"},
        soft_glossary_details=soft,
    )
    prompt = PromptBuilder.build_single_prompt(
        type("Node", (), {"content": "Interest on the loan increased."})(), context,
    )
    assert "PREFERRED / SOFT TERMS" in prompt
    assert "Never use a regex replacement to force a soft term" in prompt
    assert "finance/loan" in prompt


def test_legacy_locked_flag_maps_to_hard_or_soft_and_domain_resolves_polysemy():
    db_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(db_engine)
    db = sessionmaker(bind=db_engine)()
    try:
        db.add_all([
            GlossaryModel(id="hard", project_id="p", source_term="OpenAI", target_term="OpenAI", locked=True, domain="GENERAL"),
            GlossaryModel(id="finance-interest", project_id="p", source_term="interest", target_term="tiền lãi", locked=False, domain="FINANCE", lock_level="SOFT"),
            GlossaryModel(id="general-interest", project_id="p", source_term="interest", target_term="sự quan tâm", locked=False, domain="GENERAL", lock_level="SOFT"),
        ])
        db.commit()

        finance = GlossaryService.get_contextual_glossary(db, "p", domain="FINANCE")
        general = GlossaryService.get_contextual_glossary(db, "p", domain="GENERAL")
        assert {item["preferred_target"] for item in finance if item["source_term"] == "interest"} == {"tiền lãi"}
        assert {item["preferred_target"] for item in general if item["source_term"] == "interest"} == {"sự quan tâm"}
        assert GlossaryService.get_locked_glossary_map(db, "p")["OpenAI"] == "OpenAI"
        assert all(item["lock_level"] == "SOFT" for item in finance if item["source_term"] == "interest")
    finally:
        db.close()


def test_context_assembler_exposes_hard_and_soft_details_only_when_relevant():
    class Node:
        content = "The margin increased."

    context = ContextAssembler.assemble_context(
        [Node()],
        DocumentProfiler.fallback_profile("FINANCE", Node.content),
        ChapterMemory(), [], {}, ModelCapabilities(), [],
        soft_glossary={"margin": "biên lợi nhuận", "unused": "không dùng"},
        glossary_entries=[{
            "source_term": "margin", "preferred_target": "biên lợi nhuận",
            "allowed_variants": ["biên"], "lock_level": "SOFT",
        }],
    )
    assert context.soft_glossary == {"margin": "biên lợi nhuận"}
    assert context.soft_glossary_details[0]["allowed_variants"] == ["biên"]
    assert "unused" not in context.soft_glossary
