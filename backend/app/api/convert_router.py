"""API chuyển thẳng PDF sang EPUB/MOBI cho KOReader trên Kindle cũ (không qua luồng dịch)."""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.services.ebook.devices import DEFAULT_DEVICE, DEVICE_PROFILES, list_devices
from app.services.ebook.jobs import conversion_jobs
from app.services.ebook.pdf_to_ebook import SUPPORTED_FORMATS, ConversionOptions, calibre_executable
from app.services.extractor.ocr_engine import OCREngine

router = APIRouter(prefix="/api/convert", tags=["Convert"])

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB
_MEDIA_TYPES = {"epub": "application/epub+zip", "mobi": "application/x-mobipocket-ebook"}


@router.get("/options")
def conversion_options():
    return {
        "formats": [
            {"key": "epub", "label": "EPUB", "recommended": True,
             "description": "Khuyên dùng cho KOReader: giữ mục lục nhiều cấp, ảnh, chữ nghiêng/đậm, tự co giãn chữ."},
            {"key": "mobi", "label": "MOBI (Kindle cũ)",
             "description": "MOBI 6 cổ điển cho trình đọc gốc của Kindle; KOReader cũng mở được."},
        ],
        "devices": list_devices(),
        "default_device": DEFAULT_DEVICE,
        "calibre_available": bool(calibre_executable()),
        "tesseract_available": OCREngine().is_available,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@router.post("")
async def start_conversion(
    file: UploadFile = File(...),
    output_format: str = Form("epub"),
    device: str = Form(DEFAULT_DEVICE),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    include_images: bool = Form(True),
    cover: bool = Form(True),
    mobi_engine: str = Form("auto"),
):
    filename = Path(file.filename or "book.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Chỉ nhận file PDF (.pdf).")
    fmt = (output_format or "").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Định dạng xuất phải là EPUB hoặc MOBI.")
    if device not in DEVICE_PROFILES:
        raise HTTPException(status_code=400, detail=f"Dòng máy '{device}' không có trong danh sách.")
    if language and language.strip().lower() in ("auto", ""):
        language = None

    options = ConversionOptions(
        output_format=fmt, device=device, title=title, author=author, language=language,
        include_images=include_images, cover=cover, mobi_engine=mobi_engine, source_name=filename,
    )
    job = conversion_jobs.create(filename, options)
    source = conversion_jobs.source_path(job)
    written = 0
    try:
        with open(source, "wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File PDF quá lớn (tối đa 1 GB).")
                fh.write(chunk)
        with open(source, "rb") as fh:
            if not fh.read(1024).lstrip().startswith(b"%PDF"):
                raise HTTPException(status_code=400, detail="File tải lên không phải PDF hợp lệ.")
    except BaseException:
        conversion_jobs.discard(job.id)
        raise
    conversion_jobs.submit(job, options)
    return job.public()


@router.get("/jobs")
def list_conversions(limit: int = 20):
    return [job.public() for job in conversion_jobs.list(limit=max(1, min(limit, 100)))]


@router.get("/jobs/{job_id}")
def conversion_status(job_id: str):
    job = conversion_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy lượt chuyển đổi.")
    return job.public()


@router.get("/jobs/{job_id}/download")
def download_conversion(job_id: str):
    job = conversion_jobs.get(job_id)
    path = conversion_jobs.output_path(job) if job else None
    if not job or job.status != "done" or not path:
        raise HTTPException(status_code=404, detail="File kết quả không còn (đã quá hạn hoặc bị xóa).")
    return FileResponse(path=str(path), filename=path.name,
                        media_type=_MEDIA_TYPES.get(job.output_format, "application/octet-stream"))


@router.delete("/jobs/{job_id}")
def delete_conversion(job_id: str):
    if not conversion_jobs.delete(job_id):
        raise HTTPException(status_code=409, detail="Không xóa được (đang chạy hoặc không tồn tại).")
    return {"deleted": job_id}
