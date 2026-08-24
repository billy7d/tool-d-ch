import pytest
import uuid
from app.db.engine import get_project_db
from app.db.models import ProjectModel, NodeModel
from app.services.translation.worker import translation_worker


def test_crash_recovery_resets_translating_nodes():
    project_id = f"test_crash_{uuid.uuid4().hex[:8]}"
    db = get_project_db(project_id)

    try:
        proj = ProjectModel(id=project_id, title="Crash Test Project")
        db.add(proj)
        db.commit()

        # Simulate nodes stuck in TRANSLATING before a crash
        n1 = NodeModel(id=f"n1_{project_id}", project_id=project_id, content="Text 1", status="TRANSLATING")
        n2 = NodeModel(id=f"n2_{project_id}", project_id=project_id, content="Text 2", status="TRANSLATED", translated_content="Dịch 2")
        n3 = NodeModel(id=f"n3_{project_id}", project_id=project_id, content="Text 3", status="PENDING")

        db.add_all([n1, n2, n3])
        db.commit()

        # Execute recovery
        translation_worker.recover_crashed_jobs(project_id)

        # Verify
        db.refresh(n1)
        db.refresh(n2)
        db.refresh(n3)

        assert n1.status == "PENDING"  # Recovered
        assert n2.status == "TRANSLATED" # Preserved completed translation
        assert n3.status == "PENDING"
    finally:
        db.close()
