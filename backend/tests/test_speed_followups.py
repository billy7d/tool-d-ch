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
