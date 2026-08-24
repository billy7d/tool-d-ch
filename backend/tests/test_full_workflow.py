import pytest
import io
from fastapi.testclient import TestClient
from app.main import app
from app.db.engine import init_global_db, get_project_db
from app.services.translation.worker import translation_worker
from app.services.translation.mock_provider import MockProvider

client = TestClient(app)


def test_full_end_to_end_workflow():
    init_global_db()

    # 1. Step 1: Create Project
    res = client.post("/api/projects", json={
        "title": "E2E Test Book",
        "source_language": "en",
        "target_language": "vi",
        "translation_mode": "NATURAL",
        "document_type": "GENERAL",
        "selected_model": "mock"
    })
    assert res.status_code == 200
    proj_id = res.json()["id"]

    # 2. Step 1: Upload Document
    sample_content = (
        "# CHAPTER 1: The Foundations\n\n"
        "Investing is not a sprint, it is a marathon.\n\n"
        "## 1.1 Risk and Return\n\n"
        "High returns always accompany high uncertainty."
    ).encode("utf-8")

    res_upload = client.post(
        f"/api/projects/{proj_id}/documents",
        files={"file": ("foundations.txt", io.BytesIO(sample_content), "text/plain")}
    )
    assert res_upload.status_code == 200

    # 3. Step 2: Trigger Analysis
    from app.api.analyze_router import _run_analysis_pipeline
    _run_analysis_pipeline(proj_id)

    # 4. Step 3: Check Structure
    res_struct = client.get(f"/api/projects/{proj_id}/structure")
    assert res_struct.status_code == 200
    struct_data = res_struct.json()
    assert len(struct_data["chapters"]) >= 1
    chapter = struct_data["chapters"][0]
    assert len(chapter["nodes"]) >= 2

    # Confirm structure
    res_confirm = client.post(f"/api/projects/{proj_id}/structure/confirm", json={"confirmed": True})
    assert res_confirm.status_code == 200

    # 5. Step 4: Add Glossary Term
    res_glossary = client.post(f"/api/projects/{proj_id}/glossary", json={
        "source_term": "marathon",
        "target_term": "cuộc đua marathon",
        "locked": True
    })
    assert res_glossary.status_code == 200

    # 6. Step 5: Translate with Mock Provider
    mock_provider = MockProvider()
    translation_worker.translate_project_sync(proj_id, provider=mock_provider)

    # Verify nodes translated
    res_struct_after = client.get(f"/api/projects/{proj_id}/structure")
    struct_after = res_struct_after.json()
    translated_nodes = [n for ch in struct_after["chapters"] for n in ch["nodes"] if n.get("translated_content")]
    assert len(translated_nodes) >= 2

    # 7. Step 6: QA Audit
    res_qa = client.post(f"/api/projects/{proj_id}/qa/run")
    assert res_qa.status_code == 200

    # 8. Step 7 & 8: Export PDF & EPUB
    res_export_epub = client.post(f"/api/projects/{proj_id}/export", json={"format": "epub"})
    assert res_export_epub.status_code == 200

    res_export_pdf = client.post(f"/api/projects/{proj_id}/export", json={"format": "pdf"})
    assert res_export_pdf.status_code == 200
