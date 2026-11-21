"""Hàng đợi chuyển PDF -> ebook chạy nền, mỗi lần một file (đọc PDF dài tốn RAM và CPU).

Trạng thái lưu ra job.json cạnh file kết quả để vẫn tải lại được sau khi server khởi động lại
(start.bat chạy uvicorn --reload). Thư mục cũ hơn MAX_AGE_HOURS bị xóa khi có lượt chuyển mới.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.services.ebook.pdf_to_ebook import ConversionOptions, convert_pdf_to_ebook

MAX_AGE_HOURS = 24
MAX_KEPT_JOBS = 30


@dataclass
class ConversionJob:
    id: str
    filename: str
    output_format: str
    device: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    stage: str = "Đang chờ"
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    output_name: Optional[str] = None
    result: Optional[dict] = None

    def public(self) -> dict:
        data = asdict(self)
        data["progress"] = round(self.progress * 100, 1)
        data["download_url"] = f"/api/convert/jobs/{self.id}/download" if self.status == "done" else None
        if data.get("result"):
            data["result"] = {k: v for k, v in data["result"].items() if k != "output_path"}
        return data


class ConversionJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, ConversionJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-ebook")

    @property
    def root(self) -> Path:
        path = settings.DATA_DIR / "conversions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: ConversionJob) -> None:
        directory = self.job_dir(job.id)
        if directory.exists():
            (directory / "job.json").write_text(json.dumps(asdict(job), ensure_ascii=False), encoding="utf-8")

    def get(self, job_id: str) -> Optional[ConversionJob]:
        if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
            return None
        with self._lock:
            job = self._jobs.get(job_id)
        if job:
            return job
        meta = self.job_dir(job_id) / "job.json"
        if meta.exists():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                job = ConversionJob(**data)
                if job.status in ("queued", "running"):
                    job.status, job.error = "error", "Server đã khởi động lại khi đang chuyển đổi, vui lòng thử lại."
                return job
            except Exception:
                return None
        return None

    def list(self, limit: int = 20) -> List[ConversionJob]:
        jobs: Dict[str, ConversionJob] = {}
        for meta in self.root.glob("*/job.json"):
            job = self.get(meta.parent.name)
            if job:
                jobs[job.id] = job
        with self._lock:
            jobs.update(self._jobs)
        return sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    def output_path(self, job: ConversionJob) -> Optional[Path]:
        if not job.output_name:
            return None
        path = self.job_dir(job.id) / job.output_name
        return path if path.exists() else None

    def delete(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.status in ("queued", "running"):
            return False
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return True

    def discard(self, job_id: str) -> None:
        """Bỏ hẳn một lượt chưa chạy (tải lên lỗi giữa chừng)."""
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def cleanup(self) -> None:
        now = time.time()
        jobs = self.list(limit=10_000)
        finished = [j for j in jobs if j.status in ("done", "error")]
        for index, job in enumerate(finished):
            too_old = job.finished_at and now - job.finished_at > MAX_AGE_HOURS * 3600
            if too_old or index >= MAX_KEPT_JOBS:
                self.delete(job.id)
        # Thư mục mồ côi (không có job.json), ví dụ lượt tải lên bị ngắt.
        for directory in self.root.iterdir():
            if directory.is_dir() and not (directory / "job.json").exists() and directory.name not in self._jobs:
                if now - directory.stat().st_mtime > 3600:
                    shutil.rmtree(directory, ignore_errors=True)

    def create(self, filename: str, options: ConversionOptions) -> ConversionJob:
        try:
            self.cleanup()
        except Exception:
            pass
        job = ConversionJob(id=uuid.uuid4().hex, filename=filename, output_format=options.output_format,
                            device=options.device)
        self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job.id] = job
        self._save(job)
        return job

    def source_path(self, job: ConversionJob) -> Path:
        return self.job_dir(job.id) / "source.pdf"

    def submit(self, job: ConversionJob, options: ConversionOptions):
        return self._executor.submit(self._run, job, options)

    def _output_name(self, title: str, fmt: str) -> str:
        title = unicodedata.normalize("NFC", title)  # "Tư Duy" tách dấu thành "Tu_ Duy" nếu không dựng sẵn
        safe = "".join(ch if (ch.isalnum() or ch in " -_.,()") else "_" for ch in title).strip(" ._") or "sach"
        return f"{safe[:120]}.{fmt}"

    def _run(self, job: ConversionJob, options: ConversionOptions) -> None:
        directory = self.job_dir(job.id)
        work_dir = directory / "work"
        temp_output = directory / f"output.{options.output_format}"

        def report(fraction: float, message: str) -> None:
            job.progress = max(job.progress, min(1.0, fraction))
            job.stage = message

        job.status, job.stage = "running", "Đang chuyển đổi"
        self._save(job)
        try:
            result = convert_pdf_to_ebook(self.source_path(job), temp_output, work_dir, options, progress=report)
            name = self._output_name(result.title or Path(job.filename).stem, options.output_format)
            final_path = directory / name
            temp_output.replace(final_path)
            result.output_path = str(final_path)
            job.output_name = name
            job.result = result.to_dict()
            job.status, job.stage, job.progress = "done", "Hoàn tất", 1.0
        except Exception as exc:
            traceback.print_exc()
            job.status, job.stage = "error", "Lỗi"
            job.error = str(exc) or exc.__class__.__name__
        finally:
            job.finished_at = time.time()
            shutil.rmtree(work_dir, ignore_errors=True)
            self.source_path(job).unlink(missing_ok=True)  # không giữ bản PDF gốc sau khi chuyển
            self._save(job)


conversion_jobs = ConversionJobManager()
