"""Dự án mới mặc định dùng Gemma 4 khi chạy được, không thì qwen2.5:7b qua Ollama như trước."""
from app.models.schemas import ProjectCreate
from app.services.translation import model_defaults
from app.services.translation.llama_server_manager import LlamaServerManager
from app.services.translation.model_defaults import PREFERRED_MODEL, default_model


def test_gemma_is_the_default_when_its_file_and_llama_server_are_there(monkeypatch):
    monkeypatch.delenv("TOOL_DICH_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("TOOL_DICH_BACKEND", "auto")
    monkeypatch.setattr(LlamaServerManager, "available", lambda self, name: name == PREFERRED_MODEL)
    assert default_model() == PREFERRED_MODEL


def test_without_the_gemma_file_or_with_ollama_forced_the_default_stays_qwen(monkeypatch):
    monkeypatch.delenv("TOOL_DICH_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("TOOL_DICH_BACKEND", "auto")
    monkeypatch.setattr(LlamaServerManager, "available", lambda self, name: False)
    assert default_model() == "qwen2.5:7b"
    monkeypatch.setattr(LlamaServerManager, "available", lambda self, name: True)
    monkeypatch.setenv("TOOL_DICH_BACKEND", "ollama")
    assert default_model() == "qwen2.5:7b"


def test_environment_override_and_new_projects_use_the_resolved_default(monkeypatch):
    monkeypatch.setenv("TOOL_DICH_DEFAULT_MODEL", "llama3.1:8b")
    assert default_model() == "llama3.1:8b"
    assert ProjectCreate(title="Sách").selected_model == "llama3.1:8b"
    assert ProjectCreate(title="Sách", selected_model="qwen2.5:7b").selected_model == "qwen2.5:7b"


def test_a_broken_model_lookup_falls_back_to_qwen(monkeypatch):
    monkeypatch.delenv("TOOL_DICH_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("TOOL_DICH_BACKEND", "auto")

    def broken(self, name):
        raise OSError("ổ đĩa không đọc được")

    monkeypatch.setattr(LlamaServerManager, "available", broken)
    assert model_defaults.default_model() == "qwen2.5:7b"
