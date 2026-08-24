import os
import shutil
import subprocess
import requests
import psutil
from typing import Dict, Any, List
from app.config import settings
from app.models.schemas import HardwareInfoResponse


def check_ollama(ollama_url: str = settings.DEFAULT_OLLAMA_URL) -> Dict[str, Any]:
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=1.5)
        if res.status_code == 200:
            data = res.json()
            models = [m.get("name") for m in data.get("models", [])]
            return {"running": True, "models": models}
    except Exception:
        pass
    return {"running": False, "models": []}


def check_executable(name: str) -> bool:
    return shutil.which(name) is not None


def get_gpu_info() -> Dict[str, Any]:
    gpu_name = None
    vram_total_gb = None
    vram_free_gb = None
    cuda_available = False

    # Check via nvidia-smi if available
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            errors="ignore",
            timeout=2.0
        )
        lines = output.strip().split("\n")
        if lines and len(lines[0].split(",")) >= 3:
            parts = [p.strip() for p in lines[0].split(",")]
            gpu_name = parts[0]
            vram_total_gb = round(float(parts[1]) / 1024.0, 1)
            vram_free_gb = round(float(parts[2]) / 1024.0, 1)
            cuda_available = True
    except Exception:
        pass

    # Windows fallback via wmic/powershell if nvidia-smi didn't run
    if not gpu_name:
        try:
            output = subprocess.check_output(
                ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                encoding="utf-8",
                errors="ignore",
                timeout=2.0
            )
            names = [line.strip() for line in output.strip().split("\n") if line.strip()]
            if names:
                gpu_name = names[0]
        except Exception:
            pass

    return {
        "gpu_name": gpu_name,
        "vram_total_gb": vram_total_gb,
        "vram_free_gb": vram_free_gb,
        "cuda_available": cuda_available,
    }


def detect_hardware_and_env() -> HardwareInfoResponse:
    # CPU
    cpu_cores = psutil.cpu_count(logical=True) or 4
    cpu_name = "Generic CPU"
    try:
        output = subprocess.check_output(
            ["powershell", "-Command", "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"],
            encoding="utf-8",
            errors="ignore",
            timeout=2.0
        )
        name = output.strip().split("\n")[0].strip()
        if name:
            cpu_name = name
    except Exception:
        pass

    # RAM
    ram = psutil.virtual_memory()
    ram_total_gb = round(ram.total / (1024 ** 3), 1)
    ram_available_gb = round(ram.available / (1024 ** 3), 1)

    # Disk
    try:
        disk = psutil.disk_usage(str(settings.BASE_DIR))
        disk_free_gb = round(disk.free / (1024 ** 3), 1)
    except Exception:
        disk_free_gb = 50.0

    # GPU
    gpu_info = get_gpu_info()

    # Ollama
    ollama_info = check_ollama()

    # Tools
    tesseract_available = check_executable("tesseract")
    calibre_available = check_executable("ebook-convert")

    # Recommendation heuristic
    if gpu_info.get("vram_total_gb") and gpu_info["vram_total_gb"] >= 12:
        recommended_preset = "High Quality (14B-32B Models)"
    elif gpu_info.get("vram_total_gb") and gpu_info["vram_total_gb"] >= 6:
        recommended_preset = "Balanced (7B-8B Models e.g. Qwen2.5-7B)"
    elif ram_total_gb >= 16:
        recommended_preset = "Balanced CPU/GPU (3B-7B Models)"
    else:
        recommended_preset = "Low Memory (1.5B-3B Models)"

    return HardwareInfoResponse(
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        gpu_name=gpu_info.get("gpu_name"),
        vram_total_gb=gpu_info.get("vram_total_gb"),
        vram_free_gb=gpu_info.get("vram_free_gb"),
        cuda_available=gpu_info.get("cuda_available", False),
        disk_free_gb=disk_free_gb,
        ollama_running=ollama_info["running"],
        installed_models=ollama_info["models"],
        tesseract_available=tesseract_available,
        calibre_available=calibre_available,
        recommended_preset=recommended_preset,
    )
