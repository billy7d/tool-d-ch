"""Tìm, khởi động và dừng llama-server cho model đã có trong kho Ollama.

Không tải gì và không ghi gì vào kho Ollama: chỉ đọc manifest để biết file GGUF của model. Thiếu chương trình
llama-server hoặc model không dùng mẫu ChatML thì trả lời "không dùng được" và ứng dụng chạy qua Ollama như cũ.
"""
import atexit
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import requests

from app.config import settings
from app.services.translation.model_capabilities import get_model_capabilities
from app.services.translation.ollama_provider import PHYSICAL_CPU_CORES

logger = logging.getLogger(__name__)
_MODEL_LAYER = "application/vnd.ollama.image.model"
_TEMPLATE_LAYER = "application/vnd.ollama.image.template"
START_TIMEOUT_SECONDS = 240


class LlamaServerManager:
    _shared: Optional["LlamaServerManager"] = None

    def __init__(
        self, binary: Optional[Path] = None, port: Optional[int] = None, models_dir: Optional[Path] = None,
        local_models_dir: Optional[Path] = None,
    ):
        self.binary = Path(binary or settings.LLAMA_SERVER_PATH)
        self.port = int(port or settings.LLAMA_SERVER_PORT)
        self.models_dir = Path(models_dir or os.environ.get("OLLAMA_MODELS") or (Path.home() / ".ollama" / "models"))
        self.local_models_dir = Path(local_models_dir or settings.LOCAL_MODELS_DIR)
        self.url = f"http://127.0.0.1:{self.port}"
        self.slots = 0
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @classmethod
    def shared(cls) -> "LlamaServerManager":
        if cls._shared is None:
            cls._shared = cls()
            atexit.register(cls._shared.stop)
        return cls._shared

    def _layers(self, model_name: str) -> List[dict]:
        name, _, tag = (model_name or "").partition(":")
        manifest = self.models_dir / "manifests" / "registry.ollama.ai" / "library" / name / (tag or "latest")
        try:
            return list(json.loads(manifest.read_text(encoding="utf-8")).get("layers") or [])
        except Exception:
            return []

    def _blob(self, model_name: str, media_type: str) -> Optional[Path]:
        for layer in self._layers(model_name):
            if layer.get("mediaType") == media_type:
                path = self.models_dir / "blobs" / str(layer.get("digest", "")).replace(":", "-")
                return path if path.exists() else None
        return None

    def local_gguf(self, model_name: str) -> Optional[Path]:
        """File GGUF tải riêng có tên đúng bằng tên model (bỏ đuôi .gguf, không phân biệt hoa thường)."""
        wanted = (model_name or "").strip().casefold()
        if not wanted or not self.local_models_dir.is_dir():
            return None
        for path in sorted(self.local_models_dir.rglob("*.gguf")):
            if path.stem.casefold() == wanted and not path.stem.casefold().startswith("mmproj"):
                return path
        return None

    def local_models(self) -> List[str]:
        """Tên các model GGUF tải riêng dùng được (có llama-server), để giao diện cho chọn cùng model của Ollama."""
        if not self.binary.exists() or not self.local_models_dir.is_dir():
            return []
        from app.services.translation.model_defaults import PREFERRED_MODEL

        names = {path.stem for path in self.local_models_dir.rglob("*.gguf") if not path.stem.casefold().startswith("mmproj")}
        # Model khuyên dùng đứng đầu danh sách chọn, phần còn lại theo tên.
        return sorted(names, key=lambda name: (name != PREFERRED_MODEL, name.casefold()))

    def gguf_path(self, model_name: str) -> Optional[Path]:
        return self.local_gguf(model_name) or self._blob(model_name, _MODEL_LAYER)

    def uses_chatml(self, model_name: str) -> bool:
        """Model của Ollama: provider tự ghép ChatML để có đúng chuỗi token như Ollama; model mẫu khác chạy qua Ollama."""
        template = self._blob(model_name, _TEMPLATE_LAYER)
        try:
            return bool(template) and "<|im_start|>" in template.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

    def uses_model_template(self, model_name: str) -> bool:
        """File GGUF tải riêng: prompt ghép bằng mẫu hội thoại nằm trong chính file đó (llama-server /apply-template)."""
        return self.local_gguf(model_name) is not None

    def available(self, model_name: str) -> bool:
        if not self.binary.exists():
            return False
        if self.uses_model_template(model_name):
            return True
        return self.gguf_path(model_name) is not None and self.uses_chatml(model_name)

    def _healthy(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=2.0).status_code == 200
        except Exception:
            return False

    def _serving(self, gguf: Path) -> bool:
        """Máy chủ còn sống từ lần chạy trước chỉ dùng lại khi nó đang chạy đúng file model này."""
        try:
            props = requests.get(f"{self.url}/props", timeout=3.0).json()
        except Exception:
            return False
        if gguf.name not in str(props.get("model_path", "")):
            return False
        self.slots = int(props.get("total_slots") or 1)
        return True

    @staticmethod
    def _release_ollama(model_name: str) -> None:
        """Ollama còn giữ model trong VRAM thì GPU 6GB không đủ chỗ cho bản thứ hai."""
        try:
            requests.post(f"{settings.DEFAULT_OLLAMA_URL}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=5.0)
        except Exception:
            pass

    def _command(self, gguf: Path, model_name: str, slots: int) -> List[str]:
        capabilities = get_model_capabilities(model_name)
        context = min(capabilities.context_window, capabilities.recommended_context_window)
        command = [
            str(self.binary), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(self.port),
            # --fit off: không để llama-server tự thu nhỏ context hay đẩy lớp về CPU; thiếu VRAM thì thử ít slot hơn.
            "-ngl", "all", "--fit", "off", "-c", str(context * slots), "-np", str(slots),
            "-t", str(PHYSICAL_CPU_CORES), "--no-webui",
        ]
        if self.uses_model_template(model_name):
            # Model có chế độ suy nghĩ thì tắt: mọi lệnh gọi của pipeline chờ câu trả lời trực tiếp, có hạn token.
            command += ["--reasoning", "off"]
        return command

    def _start(self, gguf: Path, model_name: str, slots: int) -> bool:
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = settings.DATA_DIR / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, TMP=str(temp_dir), TEMP=str(temp_dir))
        with open(settings.LOGS_DIR / "llama-server.log", "ab") as log:
            self._process = subprocess.Popen(
                self._command(gguf, model_name, slots), cwd=str(self.binary.parent), env=environment,
                stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        deadline = time.time() + START_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self._process.poll() is not None:
                self._process = None
                return False
            if self._healthy():
                self.slots = slots
                return True
            time.sleep(1.0)
        self.stop()
        return False

    def ensure_running(self, model_name: str) -> bool:
        with self._lock:
            gguf = self.gguf_path(model_name)
            if not gguf or not self.available(model_name):
                return False
            if self._healthy():
                if self._serving(gguf):
                    return True
                if self._process is None:
                    logger.warning("Cổng %s đang có máy chủ chạy model khác; dùng Ollama.", self.port)
                    return False
                self.stop()
            self._release_ollama(model_name)
            if self.uses_model_template(model_name):
                self._release_ollama(settings.DEFAULT_MODEL)
            # Hai slot: lệnh có prompt riêng không xóa cache của lượt dịch. Không đủ VRAM thì còn một slot.
            for slots in (2, 1):
                if self._start(gguf, model_name, slots):
                    return True
                logger.warning("llama-server không khởi động được với %s slot.", slots)
            return False

    def stop(self) -> None:
        process, self._process = self._process, None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
