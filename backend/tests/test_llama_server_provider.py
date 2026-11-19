"""llama-server nhận đúng prompt của OllamaProvider, giữ cache theo slot và không cho model viết bộ chữ lạ."""
import json
from pathlib import Path

import pytest
import requests

import app.services.translation.llama_server_manager as manager_module
from app.services.translation.llama_server_manager import LlamaServerManager
from app.services.translation.llama_server_provider import (
    OLLAMA_DEFAULT_SAMPLING,
    LlamaServerProvider,
    chatml_prompt,
    script_ban_grammar,
)
from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.provider_base import SEMANTIC_CRITIC_SCHEMA
from app.services.translation.worker import TranslationWorker

MESSAGES = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Translate: hello"}]


def test_prompt_uses_the_chatml_layout_ollama_sends_for_qwen():
    assert chatml_prompt(MESSAGES) == (
        "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nTranslate: hello<|im_end|>\n<|im_start|>assistant\n"
    )


def test_unset_sampling_options_take_ollama_defaults_not_llama_server_defaults():
    body, _ = LlamaServerProvider().completion_request({"messages": MESSAGES, "options": {"temperature": 0.0, "num_predict": 450}})
    assert body["n_predict"] == 450 and body["temperature"] == 0.0
    assert {key: body[key] for key in ("top_k", "top_p", "min_p", "repeat_penalty", "repeat_last_n")} == {
        "top_k": 40, "top_p": 0.9, "min_p": 0.0, "repeat_penalty": 1.1, "repeat_last_n": 64,
    }
    assert set(OLLAMA_DEFAULT_SAMPLING) <= set(body)


def test_explicit_sampling_options_are_kept_and_gpu_options_are_left_to_the_server():
    provider = LlamaServerProvider()
    options = provider._options("qwen2.5:7b", 0.15, 300, top_p=0.85, repeat_penalty=1.05)
    assert "num_gpu" not in options and "num_thread" not in options
    body, _ = provider.completion_request({"messages": MESSAGES, "options": options})
    assert (body["top_p"], body["repeat_penalty"], body["temperature"]) == (0.85, 1.05, 0.15)
    assert body["cache_prompt"] is True


def test_calls_sharing_the_translation_prefix_keep_their_own_cache_slot():
    provider = LlamaServerProvider()
    provider.remember_prompt_prefix("SYS")
    other = [{"role": "system", "content": "Build structured source chapter memory"}, MESSAGES[1]]
    assert provider.completion_request({"messages": MESSAGES})[1] == 0
    assert provider.completion_request({"messages": other})[1] == 1


def test_single_slot_server_gets_every_call_on_slot_zero():
    provider = LlamaServerProvider(slots=1)
    other = [{"role": "system", "content": "another prompt"}, MESSAGES[1]]
    assert provider.completion_request({"messages": other})[0]["id_slot"] == 0


def test_scripts_absent_from_the_prompt_are_banned_from_the_output():
    grammar = script_ban_grammar("Translate: hello")
    assert grammar.startswith("root ::= [^") and "\\u4E00-\\u9FFF" in grammar and "\\u0400-\\u052F" in grammar


def test_script_present_in_the_source_may_still_be_written():
    grammar = script_ban_grammar("The character 道 means the way.")
    assert "\\u4E00-\\u9FFF" not in grammar and "\\u0400-\\u052F" in grammar
    assert script_ban_grammar("道 한 д") is None


def test_plain_text_calls_carry_the_ban_and_structured_calls_carry_their_schema():
    provider = LlamaServerProvider()
    plain, _ = provider.completion_request({"messages": MESSAGES})
    assert "grammar" in plain and "json_schema" not in plain
    structured, _ = provider.completion_request({"messages": MESSAGES, "format": SEMANTIC_CRITIC_SCHEMA})
    assert structured["json_schema"] == SEMANTIC_CRITIC_SCHEMA and "grammar" not in structured
    loose, _ = provider.completion_request({"messages": MESSAGES, "format": "json"})
    assert loose["json_schema"] == {"type": "object"}
    assert "grammar" not in LlamaServerProvider(ban_foreign_scripts=False).completion_request({"messages": MESSAGES})[0]


class _FakeResponse:
    status_code = 200
    text = "ok"

    def __init__(self, content):
        self._content = content

    def json(self):
        return {
            "content": self._content, "tokens_evaluated": 1500,
            "timings": {"prompt_n": 160, "cache_n": 1340, "prompt_ms": 1000.0, "predicted_n": 40, "predicted_ms": 900.0},
        }


@pytest.fixture
def sent(monkeypatch):
    calls = []

    def fake_post(self, url, data=None, json=None, **kwargs):
        calls.append((url, json))
        return _FakeResponse('{"status":"PASS","score":1,"checks":{},"errors":[]}' if "json_schema" in json else "Xin chào.")

    monkeypatch.setattr(requests.Session, "post", fake_post)
    return calls


def test_translation_goes_to_the_completion_endpoint_with_the_same_prompt_text(sent):
    provider = LlamaServerProvider(base_url="http://127.0.0.1:1")
    assert provider.translate_single("Hello.", "SYS", user_prompt="Translate: hello") == "Xin chào."
    url, body = sent[0]
    assert url == "http://127.0.0.1:1/completion" and body["prompt"] == chatml_prompt(MESSAGES)
    assert body["top_p"] == 0.85 and body["repeat_penalty"] == OllamaProvider.VIETNAMESE_REPEAT_PENALTY


def test_semantic_critic_keeps_its_schema_and_its_answer_is_parsed(sent):
    provider = LlamaServerProvider(base_url="http://127.0.0.1:1")
    verdict = provider.review_semantic_fidelity("Hello.", "Xin chào.", {}, {}, "GENERAL")
    assert verdict["status"] == "PASS" and sent[0][1]["json_schema"] == SEMANTIC_CRITIC_SCHEMA


def test_answer_reports_how_much_of_the_prompt_came_from_cache(sent):
    provider = LlamaServerProvider(base_url="http://127.0.0.1:1")
    body = provider.session.post("http://127.0.0.1:1/api/chat", json={"messages": MESSAGES}).json()
    assert body["message"]["content"] == "Xin chào."
    assert (body["prompt_eval_count"], body["cached_tokens"], body["eval_count"]) == (1500, 1340, 40)


def _ollama_store(root, template="<|im_start|>system"):
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "sha256-aaa").write_bytes(b"GGUF")
    (blobs / "sha256-bbb").write_text(template, encoding="utf-8")
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5" / "7b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:aaa"},
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:bbb"},
    ]}), encoding="utf-8")
    return root


def _manager(tmp_path, template="<|im_start|>system", binary=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    exe = tmp_path / "llama-server.exe"
    if binary:
        exe.write_bytes(b"")
    return LlamaServerManager(
        binary=exe, port=1, models_dir=_ollama_store(tmp_path / "models", template), local_models_dir=tmp_path / "local",
    )


def test_model_file_is_found_through_the_ollama_manifest(tmp_path):
    manager = _manager(tmp_path)
    assert manager.gguf_path("qwen2.5:7b").name == "sha256-aaa" and manager.available("qwen2.5:7b")
    assert manager.gguf_path("gemma3:4b") is None and not manager.available("gemma3:4b")


def test_missing_binary_or_non_chatml_model_is_not_available(tmp_path):
    assert not _manager(tmp_path / "a", binary=False).available("qwen2.5:7b")
    assert not _manager(tmp_path / "b", template="<start_of_turn>user").available("qwen2.5:7b")


def test_downloaded_gguf_is_found_by_file_name_and_runs_with_its_own_template(tmp_path):
    manager = _manager(tmp_path, template="<start_of_turn>user")
    local = tmp_path / "local" / "gemma4"
    local.mkdir(parents=True)
    (local / "gemma-4-E4B-it-Q4_K_M.gguf").write_bytes(b"GGUF")
    (local / "mmproj-F16.gguf").write_bytes(b"GGUF")
    assert manager.gguf_path("gemma-4-e4b-it-q4_k_m") == local / "gemma-4-E4B-it-Q4_K_M.gguf"
    assert manager.available("gemma-4-E4B-it-Q4_K_M") and manager.uses_model_template("gemma-4-E4B-it-Q4_K_M")
    assert manager.local_gguf("mmproj-F16") is None and not manager.available("qwen2.5:7b")
    assert manager.local_models() == ["gemma-4-E4B-it-Q4_K_M"]
    (local / "Another-7B-Q4_K_M.gguf").write_bytes(b"GGUF")
    assert manager.local_models() == ["gemma-4-E4B-it-Q4_K_M", "Another-7B-Q4_K_M"]
    command = manager._command(local / "gemma-4-E4B-it-Q4_K_M.gguf", "gemma-4-E4B-it-Q4_K_M", 2)
    assert command[-2:] == ["--reasoning", "off"]
    assert "--reasoning" not in _manager(tmp_path / "q")._command(Path("q.gguf"), "qwen2.5:7b", 2)


def test_downloaded_model_prompt_is_rendered_by_the_server_template(tmp_path, sent, monkeypatch):
    import app.services.translation.llama_server_provider as provider_module

    manager = _manager(tmp_path)
    local = tmp_path / "local"
    local.mkdir(parents=True)
    (local / "Hy-MT2-7B-Q4_K_M.gguf").write_bytes(b"GGUF")
    rendered = []

    class _Rendered:
        def raise_for_status(self):
            return None

        def json(self):
            return {"prompt": "<hy>SYS|Translate: hello<assistant>"}

    def fake_apply(url, json=None, **kwargs):
        rendered.append((url, json))
        return _Rendered()

    monkeypatch.setattr(provider_module.requests, "post", fake_apply)
    provider = LlamaServerProvider(default_model="Hy-MT2-7B-Q4_K_M", manager=manager)
    manager.ensure_running = lambda model: True
    manager.slots = 2
    assert provider.translate_single("Hello.", "SYS", user_prompt="Translate: hello") == "Xin chào."
    assert rendered[0][0].endswith("/apply-template") and rendered[0][1]["messages"] == MESSAGES
    assert sent[0][1]["prompt"] == "<hy>SYS|Translate: hello<assistant>"
    assert not LlamaServerProvider(default_model="qwen2.5:7b", manager=manager).use_model_template


def test_worker_falls_back_to_ollama_when_llama_server_cannot_be_used(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOL_DICH_BACKEND", "auto")
    monkeypatch.setattr(LlamaServerManager, "_shared", _manager(tmp_path / "a", binary=False))
    assert type(TranslationWorker().get_provider("qwen2.5:7b")) is OllamaProvider
    monkeypatch.setattr(LlamaServerManager, "_shared", _manager(tmp_path / "b"))
    assert isinstance(TranslationWorker().get_provider("qwen2.5:7b"), LlamaServerProvider)
    monkeypatch.setenv("TOOL_DICH_BACKEND", "ollama")
    assert type(TranslationWorker().get_provider("qwen2.5:7b")) is OllamaProvider


def test_server_is_started_by_the_first_call_not_by_creating_the_provider(tmp_path, sent):
    manager = _manager(tmp_path)
    started = []
    manager.ensure_running = lambda model: started.append(model) or True
    manager.slots = 1
    provider = LlamaServerProvider(default_model="qwen2.5:7b", manager=manager)
    assert provider.health_check() and started == []
    provider.translate_single("Hello.", "SYS", user_prompt="Translate: hello")
    assert started == ["qwen2.5:7b"] and provider.slots == 1


def test_failed_start_surfaces_as_an_error(tmp_path):
    manager = _manager(tmp_path)
    manager.ensure_running = lambda model: False
    provider = LlamaServerProvider(default_model="qwen2.5:7b", manager=manager)
    with pytest.raises(RuntimeError):
        provider.translate_single("Hello.", "SYS", user_prompt="Translate: hello")


class _Reply:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_running_server_is_reused_only_when_it_serves_the_same_model(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    props = {"model_path": "C:/models/blobs/sha256-aaa", "total_slots": 2}
    monkeypatch.setattr(manager_module.requests, "get", lambda url, timeout=0: _Reply(200, props))
    monkeypatch.setattr(manager, "_start", lambda *args: pytest.fail("không được khởi động thêm tiến trình"))
    assert manager.ensure_running("qwen2.5:7b") and manager.slots == 2
    props["model_path"] = "C:/models/other.gguf"
    assert not manager.ensure_running("qwen2.5:7b")


def test_start_retries_with_one_slot_when_two_do_not_fit(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    tried = []
    monkeypatch.setattr(manager_module.requests, "get", lambda url, timeout=0: _Reply(503))
    monkeypatch.setattr(manager_module.requests, "post", lambda *args, **kwargs: _Reply(200))
    monkeypatch.setattr(manager, "_start", lambda gguf, model, slots: tried.append(slots) or slots == 1)
    assert manager.ensure_running("qwen2.5:7b") and tried == [2, 1]


def test_speculated_translation_is_reused_only_for_the_identical_request(sent):
    provider = LlamaServerProvider(base_url="http://127.0.0.1:1")
    assert provider.speculate_translation("Hello.", "SYS", {}, "qwen2.5:7b", 0.15, "Translate: hello")
    assert provider.translate_single("Hello.", "SYS", model="qwen2.5:7b", temperature=0.15, user_prompt="Translate: hello") == "Xin chào."
    assert len(sent) == 1 and provider.speculation_hits == 1 and provider._speculation is None


def test_different_prompt_ignores_the_speculation_and_calls_the_model(sent):
    provider = LlamaServerProvider(base_url="http://127.0.0.1:1")
    provider.speculate_translation("Hello.", "SYS", {}, "qwen2.5:7b", 0.15, "Translate: hello")
    provider._speculation["thread"].join()
    provider.translate_single("Hello.", "SYS", model="qwen2.5:7b", temperature=0.15, user_prompt="Translate: hello, with new context")
    assert len(sent) == 2 and provider.speculation_hits == 0
    assert sent[1][1]["prompt"] == chatml_prompt([MESSAGES[0], {"role": "user", "content": "Translate: hello, with new context"}])


def test_single_slot_server_never_speculates(sent):
    provider = LlamaServerProvider(base_url="http://127.0.0.1:1", slots=1)
    assert not provider.speculate_translation("Hello.", "SYS", {}, "qwen2.5:7b", 0.15, "Translate: hello") and sent == []


def test_calls_made_while_the_speculation_runs_use_the_other_slot(monkeypatch):
    import threading
    release, seen = threading.Event(), []

    def slow_post(self, url, data=None, json=None, **kwargs):
        seen.append(json["id_slot"])
        if len(seen) == 1:
            release.wait(5)
        return _FakeResponse("Xin chào.")

    monkeypatch.setattr(requests.Session, "post", slow_post)
    provider = LlamaServerProvider(base_url="http://127.0.0.1:1")
    provider.remember_prompt_prefix("SYS")
    provider.speculate_translation("Hello.", "SYS", {}, "qwen2.5:7b", 0.15, "Translate: hello")
    while not seen:
        pass
    provider.translate_single("Bye.", "SYS", model="qwen2.5:7b", temperature=0.0, user_prompt="Fix this sentence")
    release.set()
    provider._speculation["thread"].join()
    assert seen == [0, 1]


CJK_RANGE = chr(92) + "u4E00-" + chr(92) + "u9FFF"


def test_foreign_script_in_system_memory_does_not_lift_the_ban():
    provider = LlamaServerProvider()
    leaked = [
        {"role": "system", "content": "CHAPTER MEMORY: summary mentions 神的存在 by mistake."},
        {"role": "user", "content": "Translate: God exists."},
    ]
    assert CJK_RANGE in provider.completion_request({"messages": leaked})[0]["grammar"]
    from_source = [
        {"role": "system", "content": "Translate faithfully."},
        {"role": "user", "content": "Translate: The character 道 means the way."},
    ]
    assert CJK_RANGE not in provider.completion_request({"messages": from_source})[0]["grammar"]


def test_chapter_memory_summary_and_tone_drop_cjk():
    from app.services.translation.context_memory import CHAPTER_MEMORY_FIELDS, validate_chapter_memory_response

    raw = {name: [] for name in CHAPTER_MEMORY_FIELDS}
    raw.update({"summary": "The author posits a God 神的存在 as the source of grace.", "tone": "reflective 反思"})
    memory = validate_chapter_memory_response(raw)
    assert memory["summary"] == "The author posits a God as the source of grace."
    assert memory["tone"] == "reflective"
