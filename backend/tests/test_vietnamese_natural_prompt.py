from app.models.canonical import TranslationMode
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.translation_signature import PROMPT_VERSION


def test_natural_prompt_prioritizes_meaning_over_english_form():
    prompt = PromptBuilder.build_system_prompt(
        document_type="GENERAL",
        translation_mode=TranslationMode.NATURAL,
    )

    assert PROMPT_VERSION == "translation-v3.3-naturalness-p0"
    assert "Preserve meaning, not English wording or grammar" in prompt
    for anti_pattern in (
        "word-for-word translation",
        "English word order",
        "excessive passive voice",
        "unnecessary nominalization",
        "awkward literal collocations",
        "redundant phrases",
        "thực hiện việc",
        "tiến hành việc",
        "một cách",
    ):
        assert anti_pattern in prompt
    assert "split or combine sentences" in prompt


def test_faithful_and_legal_profiles_keep_conservative_behavior():
    faithful = PromptBuilder.build_system_prompt(
        document_type="LEGAL",
        translation_mode=TranslationMode.FAITHFUL,
    )

    assert "Stay close to the source structure" in faithful
    assert "Preserve scope, modality, conditions" in faithful
    assert "natural vietnamese" in faithful.lower()


def test_custom_instruction_is_lower_priority_than_correctness():
    prompt = PromptBuilder.build_system_prompt(
        document_type="GENERAL",
        translation_mode=TranslationMode.NATURAL,
        custom_instructions="Hãy giữ nguyên mọi câu tiếng Anh.",
    )

    assert prompt.index("PRIORITY 1 - SEMANTIC FIDELITY") < prompt.index("USER CUSTOM INSTRUCTION")
