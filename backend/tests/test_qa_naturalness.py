from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import qa_router
from app.db.models import Base, ChapterModel, NodeModel, ProjectModel, QAIssueModel
import app.services.translation.contextual_engine as contextual_engine_module


def test_qa_run_exposes_naturalness_by_default(monkeypatch, tmp_path):
    sqlite_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(sqlite_engine)
    db = sessionmaker(bind=sqlite_engine)()
    project_id = "p0-qa-naturalness"
    db.add(ProjectModel(id=project_id, title="QA naturalness", selected_model="mock"))
    db.add(ChapterModel(id="p0-qa-chapter", project_id=project_id, title="Chapter", order_index=0))
    db.add(NodeModel(
        id="p0-qa-node",
        project_id=project_id,
        chapter_id="p0-qa-chapter",
        node_type="paragraph",
        content="The company decided to take action immediately.",
        translated_content="Công ty đã đưa ra quyết định để thực hiện hành động ngay lập tức.",
        status="TRANSLATED",
        order_index=0,
    ))
    db.commit()

    monkeypatch.setattr(qa_router, "get_project_db", lambda requested_id: db)
    monkeypatch.setattr(contextual_engine_module.settings, "PROJECTS_DIR", tmp_path / "projects")

    result = qa_router.run_qa_checks(project_id)

    assert result["naturalness_reviewed"] == 1
    assert result["naturalness_failed"] == 1
    assert result["naturalness_critic_calls"] == 1
    issue_types = {row.issue_type for row in db.query(QAIssueModel).all()}
    assert "VIETNAMESE_COLLOCATION" in issue_types
