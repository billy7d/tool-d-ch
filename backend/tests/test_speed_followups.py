"""Tối ưu tốc độ cho GTX 1660 Super 6GB + qwen2.5:7b, dựa trên số đo thật trên sách 831 đoạn."""
import pytest

from app.services.qa.semantic_critic import SemanticCriticResult, refute_false_additions
from app.services.translation.ollama_provider import PHYSICAL_CPU_CORES, OllamaProvider
from app.services.translation.prompt_builder import PromptBuilder
from app.services.translation.quality_gate import _english_residue
from app.services.translation.semantic_risk import SemanticRiskScorer
from app.services.translation.context_assembler import estimate_tokens

# Câu dài, nhiều mệnh đề nhưng không có tín hiệu ngữ nghĩa (điều kiện, tình thái, phủ định, lượng từ).
LONG = ("When I was a child my desire to win at games was strong, and the people around me paid for it, "
        "again and again, for years, in small ways and large ones, at home and at school.")
LONG_VI = ("Hồi nhỏ, tôi rất muốn thắng trong các trò chơi. Những người xung quanh phải trả giá. "
           "Điều đó kéo dài nhiều năm.")


def test_system_prompt_stays_compact():
    prompt = PromptBuilder.build_system_prompt(document_type="SELF_HELP")
    assert estimate_tokens(prompt) < 760  # trước đây ~930 token mỗi request


def test_length_alone_no_longer_requires_semantic_critic():
    long_plain = " ".join([LONG] * 3)
    assert not SemanticRiskScorer.score(long_plain, LONG_VI, document_type="SELF_HELP").requires_critic
    assert SemanticRiskScorer.score(long_plain, LONG_VI, document_type="SELF_HELP", length_risk=True).requires_critic
    # Tín hiệu ngữ nghĩa thật vẫn gọi critic.
    risky = "If you do not finish the work, you may lose the contract because the client must approve it."
    assert SemanticRiskScorer.score(risky, "x", document_type="SELF_HELP").requires_critic


def _fail(*errors):
    checks = {key: "PASS" for key in ("completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference")}
    checks["completeness"] = "FAIL"
    return SemanticCriticResult("FAIL", 0.6, list(errors), checks)


def test_addition_claims_quoting_text_that_exists_in_source_are_refuted():
    source = "Since birth and death seem to be different sides of the same coin, consider reincarnation."
    result, refuted = refute_false_additions(source, _fail(
        {"type": "SEMANTIC_ADDITION", "message": "The candidate introduces 'reincarnation' (tái sinh) which is not present in the original text."},
    ))
    assert refuted == 1 and result.status == "PASS" and result.errors == []


def test_real_additions_and_unquoted_claims_are_kept():
    source = "The house was smaller than he remembered."
    result, refuted = refute_false_additions(source, _fail(
        {"type": "SEMANTIC_ADDITION", "message": "The candidate adds 'a few years ago', not present in the original."},
        {"type": "MEANING_DRIFT", "message": "The comparison is reversed."},
    ))
    assert refuted == 0 and result.status == "FAIL" and len(result.errors) == 2


def test_capitalized_english_sentence_start_is_residue_but_titles_are_not():
    assert _english_residue("Since sinh và tử dường như chỉ là hai mặt.") == ["Since"]
    assert _english_residue("Tôi sẽ never sa thải em.") == ["never"]
    assert _english_residue("Cuốn The Road Less Traveled và bài “Journey of the Magi”.") == []
    assert _english_residue("Họ than thở mãi.") == []  # "than" cũng là từ tiếng Việt


def test_ollama_uses_all_physical_cores():
    provider = OllamaProvider()
    assert provider._options("qwen2.5:7b", 0.1, 100)["num_thread"] == PHYSICAL_CPU_CORES >= 1


def test_ollama_puts_every_layer_on_gpu_and_falls_back_when_vram_runs_out(monkeypatch):
    import requests
    from app.services.translation import ollama_provider as module

    monkeypatch.setattr(module, "_full_gpu_offload", True)
    provider = OllamaProvider()
    assert provider._options("qwen2.5:7b", 0.1, 100)["num_gpu"] == module.FULL_GPU_LAYERS
    sent = []

    def fake_post(self, url, data=None, json=None, **kwargs):
        sent.append(json)
        response = requests.Response()
        failed = "num_gpu" in json["options"]
        response.status_code = 500 if failed else 200
        response._content = (b'{"error":"CUDA error: out of memory"}' if failed else b'{"message":{"content":"ok"}}')
        return response

    monkeypatch.setattr(requests.Session, "post", fake_post)
    response = provider.session.post("http://x/api/chat", json={"options": provider._options("qwen2.5:7b", 0.1, 10)})
    assert response.status_code == 200 and len(sent) == 2 and "num_gpu" not in sent[1]["options"]
    assert "num_gpu" not in provider._options("qwen2.5:7b", 0.1, 100)


def test_letter_free_blocks_skip_the_model():
    from app.services.translation.worker import has_no_words

    assert has_no_words("***") and has_no_words("3.14") and has_no_words("[12]") and has_no_words("— 2019–2020 —")
    assert not has_no_words("(a)") and not has_no_words("Figure 3.2") and not has_no_words("  ")


def test_untranslated_content_words_are_caught_by_vietnamese_syllable_shape():
    from app.services.translation.quality_gate import TranslationQualityGate, _non_vietnamese_words

    source = "Rachel was strong, but the child kept clinging to discipline; Freud agreed."
    assert _non_vietnamese_words(source, "Rachel vẫn strong, đứa trẻ clinging đến discipline.") == [
        "strong", "clinging", "discipline",
    ]
    assert _non_vietnamese_words(source, "Strong như vậy, Freud đồng ý.") == ["Strong"]
    result = TranslationQualityGate().validate(source, "Rachel vẫn strong, nhưng đứa trẻ bám lấy kỷ luật; Freud đồng ý.", {})
    assert any(issue["code"] == "ENGLISH_RESIDUE" and "strong" in issue["message"] for issue in result.issues)


def test_vietnamese_names_loanwords_and_kept_terms_are_not_flagged():
    from app.services.translation.quality_gate import _non_vietnamese_words

    source = "Freud read How People Change. The word conscious comes from con. Machine learning and the API."
    for text in [
        "Người ta nghiêng ngả, khuya khoắt quanh khuynh hướng ấy.",
        "Freud đọc cuốn How People Change ở Rembrandt.",  # tên riêng, tựa sách viết hoa
        "Từ “conscious” có gốc Latinh.",  # chữ đang được bàn tới
        "học máy (machine learning) qua API, gửi email, 5 kg, trang ix, Kitô giáo, ôxy, iPhone.",
        r"đọc C:\x\y.txt và https://a.com/b",
    ]:
        assert _non_vietnamese_words(source, text) == [], text
    assert _non_vietnamese_words("Use the framework.", "Dùng framework này.", {"framework": "framework"}) == []


# ---------- Dịch từng đoạn với phần đầu prompt ổn định ----------

def _stable_context(document="document observations", chapter="chapter observations"):
    from app.services.translation.context_assembler import TranslationContext

    return TranslationContext(system_prompt="system rules", document_memory=document, chapter_memory=chapter)


def test_stable_prompt_is_identical_between_paragraphs_and_user_prompt_holds_only_what_changes():
    from app.models.canonical import DocumentNode, NodeStatus, NodeType
    from app.services.translation.context_memory import BilingualContextItem

    first = _stable_context()
    second = _stable_context()
    second.previous_context = [BilingualContextItem("n1", "The board met.", "Hội đồng đã họp.")]
    assert first.stable_system_prompt == second.stable_system_prompt

    node = DocumentNode(id="n2", type=NodeType.PARAGRAPH, content="The plan passed.", status=NodeStatus.PENDING, order_index=2)
    user_prompt = PromptBuilder.build_single_prompt(node, second, "Chapter 1")
    assert "The plan passed." in user_prompt and "Hội đồng đã họp." in user_prompt
    for stable_text in ("document observations", "chapter observations", "CONTEXT PRIORITY"):
        assert stable_text not in user_prompt


def _capture_ollama_payloads(monkeypatch):
    import json as json_module

    import requests

    sent = []

    def fake_post(self, url, data=None, json=None, **kwargs):
        sent.append(json)
        response = requests.Response()
        response.status_code = 200
        response._content = json_module.dumps({"message": {"content": "{}"}}).encode()
        return response

    monkeypatch.setattr(requests.Session, "post", fake_post)
    return sent


def test_critic_reuses_translation_prefix_and_forces_schema(monkeypatch):
    from app.services.translation.provider_base import REVIEW_TASK_PREAMBLE, SEMANTIC_CRITIC_SCHEMA

    sent = _capture_ollama_payloads(monkeypatch)
    provider = OllamaProvider()
    provider.remember_prompt_prefix("SHARED TRANSLATION PREFIX")
    provider.review_semantic_fidelity("Source.", "Bản dịch.", {}, {}, "GENERAL", model="qwen2.5:7b")

    messages = sent[0]["messages"]
    assert messages[0] == {"role": "system", "content": "SHARED TRANSLATION PREFIX"}
    assert messages[1]["content"].startswith(REVIEW_TASK_PREAMBLE)
    assert "semantic fidelity critic" in messages[1]["content"]
    assert sent[0]["format"] == SEMANTIC_CRITIC_SCHEMA
    # Thứ tự quyết định chất lượng: đặt errors trước status thì qwen2.5:7b chấm FAIL 32/32 lượt.
    assert list(SEMANTIC_CRITIC_SCHEMA["properties"])[:2] == ["status", "score"]
    assert set(SEMANTIC_CRITIC_SCHEMA["required"]) >= {"status", "checks", "errors"}


def test_critic_keeps_its_own_prompt_when_shared_prefix_would_overflow_context(monkeypatch):
    sent = _capture_ollama_payloads(monkeypatch)
    provider = OllamaProvider()
    provider.configure_context_window(4096)
    provider.remember_prompt_prefix("x" * 12000)
    provider.review_semantic_fidelity("Source.", "Bản dịch.", {}, {}, "GENERAL", model="qwen2.5:7b")
    assert "semantic fidelity critic" in sent[0]["messages"][0]["content"]


def test_rolling_context_always_keeps_the_paragraph_right_before_the_current_one():
    from types import SimpleNamespace

    from app.services.translation.context_memory import RollingContextService

    def row(index, words):
        return SimpleNamespace(id=f"n{index}", order_index=index, content="word " * words, translated_content="từ " * words)

    rows = [row(1, 200), row(2, 200), row(3, 200), row(4, 30)]
    kept = RollingContextService._bounded(rows, 4, 700)
    assert kept[-1].node_id == "n4"
    assert [item.node_id for item in kept] == sorted(item.node_id for item in kept)


def test_long_previous_paragraph_keeps_only_its_ending_from_a_sentence_start():
    from types import SimpleNamespace

    from app.services.translation.context_memory import ROLLING_ITEM_MAX_CHARS, RollingContextService

    source = " ".join(f"Sentence number {index} of the long paragraph." for index in range(80))
    translation = " ".join(f"Câu thứ {index} của đoạn văn dài." for index in range(80))
    row = SimpleNamespace(id="n1", order_index=1, content=source, translated_content=translation)
    item = RollingContextService._bounded([row], 4, 400)[0]
    assert len(item.source) <= ROLLING_ITEM_MAX_CHARS and len(item.translation) <= ROLLING_ITEM_MAX_CHARS
    assert item.source.startswith("Sentence number") and item.source.endswith("number 79 of the long paragraph.")
    assert item.translation.startswith("Câu thứ") and item.translation.endswith("Câu thứ 79 của đoạn văn dài.")
    short = SimpleNamespace(id="n2", order_index=2, content="Short.", translated_content="Ngắn.")
    kept = RollingContextService._bounded([short], 4, 400)[0]
    assert (kept.source, kept.translation) == ("Short.", "Ngắn.")


def test_chapter_memory_prompt_has_no_cache_metadata():
    from app.services.translation.context_memory import ChapterMemory

    memory = ChapterMemory(summary="Tóm tắt.", source_hash="a" * 64, glossary_hash="b" * 64, generated_at="2026-01-01")
    assert memory.to_prompt_dict() == {"summary": "Tóm tắt."}


def test_chapter_memory_accepts_object_notes_and_drops_chinese_items():
    from app.services.translation.context_memory import validate_chapter_memory_response

    value = validate_chapter_memory_response({
        "summary": "Chương nói về tình yêu.", "tone": "suy tư", "key_concepts": [], "important_facts": [], "style_notes": [],
        "entities": [{"source": "Jung", "preferred": "Jung"}],
        "pronoun_notes": [{"source": "I", "preferred": "tôi"}],
        "terminology": [{"source": "postulation", "preferred": "tạo hóa一神"}, {"source": "ego", "preferred": "bản ngã"}],
    })
    assert value["pronoun_notes"] == ["I -> tôi"]
    assert value["terminology"] == [{"source": "ego", "preferred": "bản ngã"}]


def test_fallback_chapter_entities_skip_function_words_and_do_not_span_lines():
    from app.services.translation.context_memory import ChapterMemoryBuilder

    names = [item["source"] for item in ChapterMemoryBuilder._entities(
        "But the theory held. And C. G. Jung wrote it.\n\nThe Prophet\n\nGive your hearts. Aniela Jaffe edited it.", {},
    )]
    assert "But" not in names and "And" not in names and "Give" not in names
    assert "Aniela Jaffe" in names
    assert not any("\n" in name for name in names)


def test_english_fused_onto_a_vietnamese_word_is_residue():
    from app.services.translation.quality_gate import _non_vietnamese_words

    source = "The unconscious part of the mind and the evolution of consciousness."
    assert _non_vietnamese_words(source, "Phần khôngscious của tâm trí và Sự Phát Triển Củasciousness.") == [
        "khôngscious", "Củasciousness",
    ]
    assert _non_vietnamese_words(source, "Nghiêng ngả, khuya khoắt, huyền thoại, thuyết phục, Mátxcơva.") == []
    assert _non_vietnamese_words("A poem by Éluard.", "Một bài thơ của Éluard.") == []


def test_word_defined_by_the_source_itself_is_not_residue():
    from app.services.translation.quality_gate import _non_vietnamese_words

    source = "The word comes from the Latin prefix con, meaning “with,” and scire, meaning “to know.”"
    assert _non_vietnamese_words(source, "Từ này gốc Latinh từ con, nghĩa là “với”, và scire, nghĩa là “biết”.") == []
    assert _non_vietnamese_words(source, "Từ này gốc Latinh và prefix có nghĩa là “với”.") == ["prefix"]


def test_citation_kept_as_is_is_not_wrong_language_but_untranslated_prose_still_is():
    from app.services.translation.language_validator import validate_target_language

    citation = "* C. G. Jung, Memories, Dreams, Reflections, Aniela Jaffe, ed. (New York: Vintage Books, 1965), p. 4."
    assert validate_target_language(citation, citation.replace("p. 4", "tr. 4")).passed
    prose = "The patient told me about her family and about the problem with her time every single week."
    assert not validate_target_language(prose, prose).passed


def test_repeated_short_sentence_is_not_a_leak_unless_the_translation_grew():
    from app.services.translation.quality_gate import TranslationQualityGate

    previous = ["Tôi đồng ý với bạn. Bạn thực sự có vấn đề về thời gian."]
    author_repeats = TranslationQualityGate().validate(
        "I agree with you. You do have a problem with time. Listen to me.",
        "Tôi đồng ý với bạn. Bạn thực sự có vấn đề về thời gian. Hãy nghe tôi.",
        {}, previous,
    )
    assert not any(issue["code"] == "CONTEXT_LEAK" for issue in author_repeats.issues)
    model_copied = TranslationQualityGate().validate(
        "Do about what?", "Bạn thực sự có vấn đề về thời gian. Về chuyện gì cơ?", {}, previous,
    )
    assert any(issue["code"] == "CONTEXT_LEAK" for issue in model_copied.issues)


def test_document_summary_is_requested_in_english(monkeypatch):
    sent = _capture_ollama_payloads(monkeypatch)
    OllamaProvider().summarize_context("Some text.", model="qwen2.5:7b")
    assert "(in English)" in sent[0]["messages"][0]["content"]
