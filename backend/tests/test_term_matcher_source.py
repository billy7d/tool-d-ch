from app.services.translation.context_assembler import ContextAssembler
from app.services.translation.glossary_validator import GlossaryValidator
from app.services.translation.term_matcher import TermMatcher

match = TermMatcher.contains_source_term


def test_capital_letters_of_the_term_must_be_capital_in_the_source():
    assert match("They ascribe the origins of grace to God.", "God")
    assert match("GOD IS LOVE", "God")
    assert not match("With an alien people clutching their gods.", "God")
    assert match("Progress in AI is fast.", "AI")
    assert not match("They said ai, ai, ai.", "AI")


def test_lowercase_term_still_matches_sentence_start_and_inflections():
    assert match("Grace is available to everyone.", "grace")
    assert match("It was so. Grace then appears.", "grace")
    assert match("the ego boundaries collapse", "ego boundaries")
    assert match("Ego boundaries collapse.", "ego boundaries")
    assert match("several therapists agreed", "therapist")


def test_lowercase_term_next_to_a_capitalized_word_is_a_proper_name():
    assert not match("* Revelations of Divine Love, Grace Warrack, ed. (New York, 1923).", "grace")
    assert not match("He spoke with Sister Grace about it.", "grace")
    assert match("Grace Warrack edited it, but grace is the theme.", "grace")


def test_target_side_matching_is_unchanged():
    assert TermMatcher.contains("Ân sủng dành cho mọi người.", "ân sủng")
    assert TermMatcher.contains("their gods", "God")


def test_prompt_and_gate_skip_terms_that_only_look_alike():
    glossary = {"God": "Chúa", "grace": "ân sủng"}
    source = "With an alien people clutching their gods. Edited by Grace Warrack."
    assert ContextAssembler.filter_relevant_glossary(source, glossary) == {}
    result = GlossaryValidator.validate(source, "Với một dân tộc xa lạ ôm chặt các vị thần của họ. Grace Warrack biên tập.", glossary)
    assert result.passed
