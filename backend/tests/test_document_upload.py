import pytest
import io
from fastapi.testclient import TestClient
from app.main import app
from app.db.engine import init_global_db

client = TestClient(app)


def test_upload_document_flow():
    init_global_db()

    # 1. Create a project
    res_proj = client.post("/api/projects", json={
        "title": "Upload Test Project",
        "source_language": "en",
        "target_language": "vi"
    })
    assert res_proj.status_code == 200
    proj_data = res_proj.json()
    project_id = proj_data["id"]

    # 2. Upload text document
    sample_text = b"# Chapter 1\n\nThis is a sample document for testing upload."
    file_payload = ("test_doc.txt", io.BytesIO(sample_text), "text/plain")

    res_upload = client.post(
        f"/api/projects/{project_id}/documents",
        files={"file": file_payload}
    )
    assert res_upload.status_code == 200
    doc_data = res_upload.json()
    assert doc_data["filename"] == "test_doc.txt"
    assert doc_data["file_format"] == "txt"
    assert doc_data["file_size_bytes"] == len(sample_text)

    # 3. List documents
    res_list = client.get(f"/api/projects/{project_id}/documents")
    assert res_list.status_code == 200
    docs = res_list.json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "test_doc.txt"
