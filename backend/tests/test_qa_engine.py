import pytest
from app.services.qa.deterministic_qa import DeterministicQA
from app.db.models import NodeModel


def test_deterministic_qa_number_mismatch():
    node = NodeModel(
        id="n1",
        content="The company grew by 45% in 2026 generating $19.99M in revenue.",
        translated_content="Công ty đã tăng trưởng trong năm 2026 và tạo ra doanh thu lớn." # Missing 45% and $19.99
    )

    issues = DeterministicQA.audit_node(node)
    issue_types = [i["issue_type"] for i in issues]
    assert "NUMBER_MISMATCH" in issue_types


def test_deterministic_qa_truncation():
    node = NodeModel(
        id="n2",
        content="This is a very comprehensive multi-sentence paragraph explaining fundamental macroeconomic principles, monetary policies, inflation targeting, and interest rate adjustments in extensive detail.",
        translated_content="Tóm tắt ngắn." # Excessive truncation
    )

    issues = DeterministicQA.audit_node(node)
    issue_types = [i["issue_type"] for i in issues]
    assert "POSSIBLE_TRUNCATION" in issue_types


def test_retranslate_all_qa_issues_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    from app.db.engine import get_project_db
    from app.db.models import ProjectModel, ChapterModel, NodeModel, QAIssueModel
    from app.db.repository import QARepository

    client = TestClient(app)
    project_id = "test_bulk_qa_fix_proj"
    db = get_project_db(project_id)
    try:
        # Clear existing
        db.query(QAIssueModel).filter(QAIssueModel.project_id == project_id).delete()
        db.query(NodeModel).filter(NodeModel.project_id == project_id).delete()
        db.query(ChapterModel).filter(ChapterModel.project_id == project_id).delete()
        db.query(ProjectModel).filter(ProjectModel.id == project_id).delete()
        db.commit()

        proj = ProjectModel(id=project_id, title="QA Bulk Project", selected_model="mock-qwen2.5:7b")
        db.add(proj)
        ch = ChapterModel(id="ch_qa", project_id=project_id, title="Chương QA", order_index=0)
        db.add(ch)
        n1 = NodeModel(
            id="n_qa_1",
            project_id=project_id,
            chapter_id="ch_qa",
            node_type="paragraph",
            content="Cash flow management is essential for investment stability.",
            translated_content="Dòng tiền... (bị lỗi)",
            order_index=0,
            status="TRANSLATED",
            version=1
        )
        db.add(n1)
        db.commit()

        qa_repo = QARepository(db)
        iss1 = qa_repo.add_issue(
            project_id=project_id,
            node_id="n_qa_1",
            issue_type="NUMBER_MISMATCH",
            severity="WARNING",
            message="Thiếu số liệu"
        )

        # Call bulk retranslate endpoint
        res = client.post(f"/api/projects/{project_id}/qa/retranslate_all_issues", json={
            "instruction": "Dịch chuẩn xác hơn"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["fixed_nodes"] == 1
        assert data["total_nodes"] == 1

        # Verify issue resolved in DB
        db.refresh(iss1)
        assert iss1.status == "RESOLVED"

    finally:
        db.close()


