"""P0 sau đợt kiểm tra: tùy chọn Ollama, bộ nhớ chương và bản nháp bị chặn."""
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.engine import get_project_db, init_global_db
from app.db.models import Base, ChapterModel, NodeModel, ProjectModel
from app.main import app
from app.services.qa.vietnamese_naturalness_critic import NATURALNESS_CHECKS
from app.services.translation.ollama_provider import CHAPTER_MEMORY_SAMPLE_CHARS, OllamaProvider
from app.services.translation.provider_base import output_token_cap
from app.services.translation.review_draft import save_review_draft
from app.services.translation.semantic_assurance import SemanticAssuranceResult, SemanticAssuranceService
from app.services.translation.translation_commit import TranslationCommitService


class _Response:
    def __init__(self, content):
        self.status_code = 200
        self._content = content
        self.text = content

    def json(self):
        return {"message": {"content": self._content}}


def _capturing_provider(content="Xin chào."):
    provider = OllamaProvider()
    sent = []

    def post(url, json=None, timeout=None):
        sent.append({"payload": json, "timeout": timeout})
        return _Response(content)

    provider.session.post = post
    return provider, sent


# ---------- Tùy chọn Ollama đồng nhất ----------

def test_every_ollama_call_sets_same_num_ctx_and_an_output_cap():
    provider, sent = _capturing_provider(json.dumps({"status": "PASS", "score": 1, "errors": [], "checks": {}}))
    provider.translate_single("Hello.", "system", model="qwen2.5:7b")
    provider.revise("Hello.", "Xin chào.", "Sửa", model="qwen2.5:7b")
    provider.summarize_context("Some text.", model="qwen2.5:7b")
    provider.build_chapter_memory("Some text.", "Chapter 1", "GENERAL", model="qwen2.5:7b")
    try:
        provider.review_semantic_fidelity("Hello.", "Xin chào.", {}, {}, "GENERAL", model="qwen2.5:7b")
    except Exception:
        pass
    contexts = {call["payload"]["options"]["num_ctx"] for call in sent}
    assert len(contexts) == 1, "num_ctx khác nhau làm Ollama phải nạp lại model"
    assert all(call["payload"]["options"].get("num_predict", 0) > 0 for call in sent)


def test_output_cap_scales_with_input_and_is_bounded():
    assert output_token_cap("short") == 256
    assert output_token_cap("x" * 800) == 464  # ~200 token nguồn * 2 + 64
    assert output_token_cap("x" * 400_000) == 2048


def test_chapter_memory_uses_short_sample_and_logs_failures(caplog):
    provider, sent = _capturing_provider("not json")
    with caplog.at_level(logging.WARNING):
        assert provider.build_chapter_memory("A" * 9000, "Chapter 7", "GENERAL", model="qwen2.5:7b") == {}
    user_message = sent[0]["payload"]["messages"][1]["content"]
    assert user_message.count("A") <= CHAPTER_MEMORY_SAMPLE_CHARS + 5
    assert sent[0]["timeout"] >= 120
    assert "Chapter memory failed" in caplog.text


# ---------- Bản nháp bị chặn ----------

@pytest.fixture()
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(ProjectModel(id="p", title="P"))
    db.add(NodeModel(id="n", project_id="p", content="Revenue rose 12%.", node_type="paragraph",
                     status="PENDING", order_index=0, version=1))
    db.commit()
    yield db
    db.close()
    engine.dispose()


def test_semantic_rejection_keeps_draft_and_commit_clears_it(memory_db):
    node = memory_db.query(NodeModel).one()
    rejected = SemanticAssuranceResult(
        "Doanh thu tăng 15%.", False, "FAIL", 0.7, "MEDIUM",
        [{"type": "MEANING_DRIFT", "severity": "ERROR", "message": "Sai số liệu."}], "sig",
    )
    SemanticAssuranceService.persist_rejected(memory_db, "p", node, rejected, "mock")
    memory_db.refresh(node)
    draft = node.node_metadata["draft_translation"]
    assert node.status == "NEEDS_REVIEW" and node.translated_content is None
    assert draft["text"] == "Doanh thu tăng 15%." and draft["reasons"] == ["MEANING_DRIFT"] and draft["stage"] == "SEMANTIC"

    TranslationCommitService.commit_validated_translation(
        memory_db, "p", node, "Doanh thu tăng 12%.",
        SimpleNamespace(style_hash="s", glossary_hash="g", prompt_version="v"), {}, "mock", "test",
    )
    memory_db.refresh(node)
    assert "draft_translation" not in (node.node_metadata or {})


def test_empty_candidate_is_not_saved_as_draft():
    node = SimpleNamespace(node_metadata={})
    save_review_draft(node, "   ", ["EMPTY_TRANSLATION"], "QUALITY_GATE")
    assert node.node_metadata == {}


def test_draft_reaches_structure_api_and_manual_save_clears_it():
    init_global_db()
    client = TestClient(app)
    project_id = client.post("/api/projects", json={"title": "Draft API", "selected_model": "mock"}).json()["id"]
    try:
        db = get_project_db(project_id)
        db.add(ChapterModel(id=f"{project_id}_ch", project_id=project_id, title="C", order_index=0))
        db.add(NodeModel(id=f"{project_id}_n", project_id=project_id, chapter_id=f"{project_id}_ch", content="Hello.", node_type="paragraph",
                         status="NEEDS_REVIEW", order_index=0, version=1,
                         node_metadata={"draft_translation": {"text": "Xin chào.", "reasons": ["X"], "stage": "SEMANTIC", "saved_at": "t"}}))
        db.commit()
        db.close()
        structure = client.get(f"/api/projects/{project_id}/structure").json()
        api_nodes = [n for ch in structure.get("chapters", []) for n in ch["nodes"]] + structure.get("unassigned_nodes", [])
        api_node = next(n for n in api_nodes if n["id"] == f"{project_id}_n")
        # Bước 6 đọc bản nháp từ metadata của node trong API structure.
        assert api_node["metadata"]["draft_translation"]["text"] == "Xin chào."
        response = client.patch(f"/api/projects/{project_id}/structure/nodes/{project_id}_n", json={"translated_content": "Chào bạn."})
        assert response.status_code == 200
        db = get_project_db(project_id)
        saved = db.query(NodeModel).filter_by(id=f"{project_id}_n").one()
        assert saved.translated_content == "Chào bạn." and "draft_translation" not in (saved.node_metadata or {})
        db.close()
    finally:
        client.delete(f"/api/projects/{project_id}")
