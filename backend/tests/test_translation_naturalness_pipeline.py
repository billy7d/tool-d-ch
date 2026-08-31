from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, ChapterModel, NodeModel, ProjectModel, TranslationMemoryModel
from app.services.translation.mock_provider import MockProvider
from app.services.translation.worker import TranslationWorker
import app.db.engine as db_engine_module
import app.services.translation.worker as worker_module
import app.services.translation.contextual_engine as contextual_engine_module


def _checks(value="PASS"):
    return {
        key: value for key in (
            "literal_calque", "word_order", "collocation", "cohesion", "pronoun_reference",
            "register", "redundancy", "nominalization", "passive_voice", "sentence_flow",
        )
    }


def _naturalness(status="PASS", score=0.96):
    if status == "PASS":
        return {"status": "PASS", "score": score, "issues": [], "checks": _checks()}
    checks = _checks()
    checks["literal_calque"] = "FAIL"
    return {
        "status": "FAIL",
        "score": score,
        "issues": [{
            "type": "LITERAL_CALQUE",
            "severity": "WARNING",
            "target_span": "đưa ra quyết định",
            "message": "Candidate còn dịch sát.",
        }],
        "checks": checks,
    }


class WorkerNaturalnessProvider(MockProvider):
    def __init__(self):
        self.naturalness_results = [_naturalness("FAIL", 0.58), _naturalness("PASS")]
        self.editorial_calls = 0

    def translate(self, blocks, system_prompt, user_prompt, model=None, temperature=0.2):
        return [{
            "node_id": block.get("id"),
            "text": "Công ty đã đưa ra quyết định để thực hiện hành động ngay lập tức.",
        } for block in blocks]

    def review_naturalness(self, *args, **kwargs):
        return self.naturalness_results.pop(0)

    def editorial_rewrite(self, **kwargs):
        self.editorial_calls += 1
        assert kwargs["source_text"] == "The company decided to take action immediately."
        return "Công ty quyết định hành động ngay."


def test_worker_saves_only_editorially_approved_candidate(monkeypatch, tmp_path):
    sqlite_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(sqlite_engine)
    db = sessionmaker(bind=sqlite_engine)()
    project_id = "p0-worker-pipeline"
    db.add(ProjectModel(id=project_id, title="P0 worker", selected_model="mock", qa_level="HIGH_QUALITY"))
    db.add(ChapterModel(id="p0-worker-chapter", project_id=project_id, title="Chapter", order_index=0))
    db.add(NodeModel(
        id="p0-worker-node",
        project_id=project_id,
        chapter_id="p0-worker-chapter",
        node_type="paragraph",
        content="The company decided to take action immediately.",
        status="PENDING",
        order_index=0,
    ))
    db.commit()

    provider = WorkerNaturalnessProvider()
    monkeypatch.setattr(worker_module, "get_project_db", lambda requested_id: db)
    monkeypatch.setattr(db_engine_module, "get_project_db", lambda requested_id: db)
    monkeypatch.setattr(contextual_engine_module.settings, "PROJECTS_DIR", tmp_path / "projects")

    TranslationWorker().translate_project_sync(project_id, model_name="mock", provider=provider)

    node = db.query(NodeModel).filter_by(id="p0-worker-node").one()
    assert node.status == "TRANSLATED"
    assert node.translated_content == "Công ty quyết định hành động ngay."
    assert provider.editorial_calls == 1
    assert db.query(TranslationMemoryModel).count() == 1
    assert db.query(TranslationMemoryModel).one().translated_text == node.translated_content
    assert "đưa ra quyết định" not in db.query(TranslationMemoryModel).one().translated_text
