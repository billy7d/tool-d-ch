from app.models.canonical import DocumentNode, NodeStatus, NodeType
from app.services.translation.context_assembler import (
    CONTEXT_TRIM_ORDER,
    NATURAL_BALANCED_CONTEXT_PRIORITY,
    ContextAssembler,
    TranslationContext,
)
from app.services.translation.context_memory import BilingualContextItem, ChapterMemory
from app.services.translation.document_profiler import DocumentProfiler
from app.services.translation.few_shot_selector import FewShotSelector
from app.services.translation.model_capabilities import ModelCapabilities
from app.services.translation.prompt_builder import PromptBuilder


def _node(text: str) -> DocumentNode:
    return DocumentNode(
        id="current", type=NodeType.PARAGRAPH, content=text,
        status=NodeStatus.PENDING, order_index=2,
    )


def test_prompt_order_keeps_source_and_high_value_context_before_distant_memory():
    previous = BilingualContextItem("previous", "The board met.", "Hội đồng đã họp.")
    context = TranslationContext(
        previous_context=[previous],
        glossary={"board": "hội đồng"},
        glossary_details=[{"source_term": "board", "preferred_target": "hội đồng", "lock_level": "HARD"}],
        style_memory_examples=[{"source_text": "The board agreed.", "approved_vi": "Hội đồng nhất trí."}],
        few_shots=[{"source": "The board met.", "target": "Hội đồng đã họp."}],
        document_memory="document observations",
        chapter_memory="chapter observations",
    )
    prompt = PromptBuilder.build_single_prompt(_node("The board approved the plan."), context, "Chapter 1")

    positions = [
        prompt.index("CURRENT SOURCE"),
        prompt.index("MANDATORY HARD TERMS"),
        prompt.index("IMMEDIATE APPROVED BILINGUAL CONTEXT"),
        prompt.index("RETRIEVED USER STYLE MEMORY"),
        prompt.index("CURATED DOMAIN EXAMPLES"),
        prompt.index("SOURCE DOCUMENT CHARACTERISTICS"),
        prompt.index("CHAPTER MEMORY"),
    ]
    assert positions == sorted(positions)
    assert "Never copy their facts, entities, numbers, or propositions" in prompt


def test_overflow_trims_distant_context_before_few_shot_and_keeps_required_layers():
    node = _node("The Federal Reserve reviewed the rate and the board approved it.")
    profile = DocumentProfiler.fallback_profile("FINANCE", node.content)
    profile.summary = "distant document detail " * 2000
    memory = ChapterMemory(summary="old chapter detail " * 1800)
    rolling = [
        BilingualContextItem(f"old-{index}", "previous source " * 80, "bản dịch trước " * 80)
        for index in range(5)
    ]
    shots = FewShotSelector.select("FINANCE", "NATURAL", "paragraph", node.content, limit=4)
    styles = [
        {"source_text": "style source " * 90, "approved_vi": "phong cách đã duyệt " * 90, "quality_status": "USER_APPROVED"}
        for _ in range(3)
    ]
    context = ContextAssembler.assemble_context(
        [node], profile, memory, rolling,
        {"Federal Reserve": "Cục Dự trữ Liên bang Mỹ", "board": "hội đồng quản trị"},
        ModelCapabilities(1400, 1400, 300, True, True),
        shots,
        system_prompt="system contract",
        output_contract="plain output",
        entity_decisions={"Federal Reserve": "Cục Dự trữ Liên bang Mỹ"},
        soft_glossary={"board": "hội đồng quản trị"},
        style_memory_examples=styles,
        glossary_entries=[
            {"source_term": "Federal Reserve", "preferred_target": "Cục Dự trữ Liên bang Mỹ", "lock_level": "HARD"},
            {"source_term": "board", "preferred_target": "hội đồng quản trị", "lock_level": "SOFT"},
        ],
    )

    assert context.glossary["Federal Reserve"] == "Cục Dự trữ Liên bang Mỹ"
    assert context.previous_context
    assert context.few_shots
    assert context.token_budget.context_tokens >= (
        context.token_budget.soft_glossary_tokens + context.token_budget.style_memory_tokens
    )
    assert context.trim_steps
    assert context.trim_steps.index("document_memory_compact_1") < context.trim_steps.index("few_shot_reduce") if "few_shot_reduce" in context.trim_steps else True


def test_emergency_overflow_is_reported_without_dropping_source_or_hard_glossary():
    node = _node("OpenAI published the report.")
    context = ContextAssembler.assemble_context(
        [node],
        DocumentProfiler.fallback_profile("GENERAL", node.content),
        ChapterMemory(),
        [],
        {"OpenAI": "OpenAI"},
        ModelCapabilities(512, 512, 64, True, True),
        [{"source": "huge " * 2000, "target": "ví dụ " * 2000}],
        system_prompt="system contract",
        output_contract="output contract",
    )
    prompt = PromptBuilder.build_single_prompt(node, context)

    assert "OpenAI" in context.glossary
    assert "OpenAI published the report." in prompt
    assert "few_shot_drop_emergency" in context.trim_steps
    assert not context.fits


def test_priority_and_trim_policy_are_explicit_and_stable():
    assert NATURAL_BALANCED_CONTEXT_PRIORITY[:5] == (
        "system_contract", "source", "hard_glossary_and_entities",
        "immediate_previous_bilingual", "user_style_memory",
    )
    assert CONTEXT_TRIM_ORDER[:4] == (
        "document_memory_compact", "chapter_memory_compact",
        "rolling_context_oldest", "entity_decisions_optional",
    )
