from app.models.canonical import DocumentNode, NodeStatus, NodeType
from app.services.translation.context_assembler import ContextAssembler, TranslationContext
from app.services.translation.context_memory import BilingualContextItem, ChapterMemory
from app.services.translation.document_profiler import DocumentProfiler
from app.services.translation.model_capabilities import ModelCapabilities
from app.services.translation.prompt_builder import PromptBuilder


def test_segment_prompt_contains_only_immediate_previous_bilingual_context():
    previous = BilingualContextItem(
        "node:segment:1",
        "Alice submitted the report.",
        "Alice đã nộp báo cáo.",
    )
    older = BilingualContextItem("node:segment:0", "The board met.", "Hội đồng đã họp.")
    context = TranslationContext(
        previous_context=[previous],
        continuity_context=previous,
        glossary={"report": "báo cáo"},
        glossary_details=[{"source_term": "report", "preferred_target": "báo cáo", "lock_level": "HARD"}],
    )
    prompt = PromptBuilder.build_segment_prompt(
        "She expected the board to review it the next day.",
        context,
        "Chapter 1",
        segment_index=2,
        segment_count=3,
    )

    assert "PREVIOUS SEGMENT SOURCE:\nAlice submitted the report." in prompt
    assert "PREVIOUS SEGMENT APPROVED VIETNAMESE:\nAlice đã nộp báo cáo." in prompt
    assert "The board met." not in prompt
    assert prompt.index("CURRENT SEGMENT") < prompt.index("PREVIOUS SEGMENT SOURCE")
    assert "Translate CURRENT SEGMENT only" in prompt


def test_context_assembler_replaces_old_rolling_history_with_current_segment():
    class Node:
        content = "She expected the board to review it the next day."

    previous = BilingualContextItem("node:segment:1", "Alice submitted the report.", "Alice đã nộp báo cáo.")
    context = ContextAssembler.assemble_context(
        [Node()],
        DocumentProfiler.fallback_profile("GENERAL", Node.content),
        ChapterMemory(),
        [BilingualContextItem("old", "old source", "bản dịch cũ"), previous],
        {"report": "báo cáo"},
        ModelCapabilities(), [],
        continuity_context=previous,
    )

    assert [item.node_id for item in context.previous_context] == ["node:segment:1"]
    assert context.continuity_context == previous


def test_merged_segments_are_still_checked_for_numbers_and_negation():
    from app.services.translation.quality_gate import TranslationQualityGate

    result = TranslationQualityGate().validate(
        "Alice did not submit the report on 12 May.",
        "Alice đã nộp báo cáo vào ngày 13 tháng 5.",
        {},
    )
    codes = {issue["code"] for issue in result.issues}
    assert "NUMBER_MISMATCH" in codes
    assert "NEGATION_LOSS" in codes
