import pytest
import io
import pymupdf
from fastapi.testclient import TestClient
from app.main import app
from app.db.engine import init_global_db

client = TestClient(app)


def test_pdf_upload_flow():
    init_global_db()

    # 1. Create a project
    res_proj = client.post("/api/projects", json={
        "title": "PDF Upload Test Project",
        "source_language": "en",
        "target_language": "vi"
    })
    assert res_proj.status_code == 200
    proj_data = res_proj.json()
    project_id = proj_data["id"]

    # 2. Create a genuine PDF in memory
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 72), "CHAPTER 1\n\nThis is a real PDF test for upload.")
    pdf_bytes = doc.tobytes()
    doc.close()

    # 3. Upload PDF
    file_payload = ("test_book.pdf", io.BytesIO(pdf_bytes), "application/pdf")
    res_upload = client.post(
        f"/api/projects/{project_id}/documents",
        files={"file": file_payload}
    )
    assert res_upload.status_code == 200
    doc_data = res_upload.json()
    assert doc_data["filename"] == "test_book.pdf"
    assert doc_data["file_format"] == "pdf"
    assert doc_data["page_count"] == 1
    assert doc_data["file_size_bytes"] == len(pdf_bytes)
