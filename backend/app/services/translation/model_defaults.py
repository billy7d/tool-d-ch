"""Model mặc định cho dự án mới.

Gemma 4 E4B (file GGUF tải riêng, chạy bằng llama-server) khi dùng được: đo 2026-11-19 trên cùng pipeline, 119 đoạn
thuộc 3 mẫu, nó nhanh hơn qwen2.5:7b 38–51% và thắng 92/112 cặp đọc mù. Thiếu file model hoặc llama-server, hay
TOOL_DICH_BACKEND=ollama, thì về qwen2.5:7b qua Ollama như trước. TOOL_DICH_DEFAULT_MODEL ghi đè cả hai.
"""
import os

from app.config import settings

PREFERRED_MODEL = "gemma-4-E4B-it-Q4_K_M"


def default_model() -> str:
    override = (os.environ.get("TOOL_DICH_DEFAULT_MODEL") or "").strip()
    if override:
        return override
    if os.environ.get("TOOL_DICH_BACKEND", "auto").lower() == "ollama":
        return settings.DEFAULT_MODEL
    try:
        from app.services.translation.llama_server_manager import LlamaServerManager

        if LlamaServerManager().available(PREFERRED_MODEL):
            return PREFERRED_MODEL
    except Exception:
        pass
    return settings.DEFAULT_MODEL
