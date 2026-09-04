from app.services.translation.few_shot_library import SUPPORTED_DOMAINS, curated_examples, library_counts
from app.services.translation.few_shot_selector import FewShotSelector


def test_curated_library_covers_every_domain_with_minimum_examples():
    counts = library_counts()
    assert set(counts) == set(SUPPORTED_DOMAINS)
    assert all(counts[domain] >= 20 for domain in SUPPORTED_DOMAINS)
    assert all(example.source and example.target and example.patterns for domain in SUPPORTED_DOMAINS for example in curated_examples(domain))


def test_selector_prefers_matching_domain_node_type_and_pattern_deterministically():
    source = "She tried to break the ice before the difficult interview began."
    first = FewShotSelector.select(
        "LITERATURE", "NATURAL", "paragraph", source_text=source, limit=4,
    )
    second = FewShotSelector.select(
        "LITERATURE", "NATURAL", "paragraph", source_text=source, limit=4,
    )

    assert first == second
    assert 2 <= len(first) <= 4
    assert all(item["domain"] == "LITERATURE" for item in first)
    assert any("idiom" in item["patterns"] for item in first)


def test_selector_honors_node_type_and_limit_without_llm_call():
    selected = FewShotSelector.select(
        "GENERAL", "NATURAL", "dialogue", source_text="She said that they could begin.", limit=20,
    )

    assert len(selected) == 4
    assert selected[0]["node_type"] == "dialogue"
    assert len(FewShotSelector.select("GENERAL", limit=2)) == 2


def test_legal_selector_uses_formal_curated_examples():
    selected = FewShotSelector.select(
        "LEGAL", "NATURAL", "paragraph",
        source_text="The supplier shall not disclose the agreement.", limit=3,
    )

    assert selected
    assert all(item["register"] == "formal" for item in selected)
    assert any("legal_obligation" in item["patterns"] for item in selected)


def test_empty_library_falls_back_safely(monkeypatch):
    monkeypatch.setattr(
        "app.services.translation.few_shot_selector.curated_examples",
        lambda _domain: [],
    )

    assert FewShotSelector.select("GENERAL", source_text="A sentence.") == []
