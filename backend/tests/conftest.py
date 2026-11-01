"""Cô lập dữ liệu của test khỏi data/ thật.

Trước đây nhiều test tạo project qua API ghi thẳng vào data/projects và global.db mà không xóa,
mỗi lần chạy suite để lại hàng chục thư mục test_crash_* và UUID. Biến môi trường phải được
đặt trước khi import app.config, nên làm ở cấp module của conftest.
"""
import os
import shutil
from pathlib import Path

_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / ".test-data"
_RUN_DIR = _TEST_DATA_ROOT / f"run-{os.getpid()}"
os.environ.setdefault("TOOL_DICH_DATA_DIR", str(_RUN_DIR))


def pytest_sessionfinish(session, exitstatus):
    if Path(os.environ["TOOL_DICH_DATA_DIR"]) != _RUN_DIR:
        return
    try:
        # SQLite trên Windows giữ file đang mở; đóng engine trước khi xóa.
        from app.db import engine as db_engine

        db_engine.global_engine.dispose()
        for cached in list(getattr(db_engine, "_project_engines", {}).values()):
            cached.dispose()
    except Exception:
        pass
    shutil.rmtree(_RUN_DIR, ignore_errors=True)
    try:
        _TEST_DATA_ROOT.rmdir()  # chỉ xóa khi đã rỗng (không có lượt chạy song song nào khác)
    except OSError:
        pass
