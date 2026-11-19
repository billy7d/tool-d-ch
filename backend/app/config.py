import os
from pathlib import Path
from pydantic import BaseModel

class AppSettings(BaseModel):
    APP_NAME: str = "Local AI Document Translator & Reflow Publisher"
    APP_VERSION: str = "1.0.0"
    HOST: str = "127.0.0.1"
    PORT: int = 8765
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    # TOOL_DICH_DATA_DIR cho phép test/benchmark dùng thư mục dữ liệu riêng, không làm bẩn data/ thật.
    DATA_DIR: Path = Path(os.environ.get("TOOL_DICH_DATA_DIR") or (BASE_DIR / "data"))
    PROJECTS_DIR: Path = DATA_DIR / "projects"
    # Từ điển Anh–Việt tùy chọn (StarDict), dùng làm nghĩa tham khảo khi sửa từ model để sót. Không phụ thuộc
    # TOOL_DICH_DATA_DIR vì đây là dữ liệu dùng chung, không phải dữ liệu của từng lần chạy.
    DICTIONARY_DIR: Path = Path(os.environ.get("TOOL_DICH_DICTIONARY_DIR") or (BASE_DIR / "data" / "dictionaries" / "en-vi"))
    FONTS_DIR: Path = BASE_DIR / "fonts"
    LOGS_DIR: Path = BASE_DIR / "logs"
    STATIC_DIR: Path = BASE_DIR / "frontend" / "dist"
    
    # AI Engine defaults
    DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
    # Model Ollama dự phòng. Dự án mới lấy model mặc định từ model_defaults.default_model() (Gemma 4 khi có file).
    DEFAULT_MODEL: str = "qwen2.5:7b"
    # llama-server (llama.cpp) chạy cùng file model của Ollama; có thì dùng, không có thì chạy qua Ollama.
    # TOOL_DICH_BACKEND: auto (mặc định) | ollama.
    LLAMA_SERVER_PATH: Path = Path(os.environ.get("TOOL_DICH_LLAMA_SERVER") or (BASE_DIR / "tools" / "llama.cpp" / "llama-server.exe"))
    LLAMA_SERVER_PORT: int = int(os.environ.get("TOOL_DICH_LLAMA_PORT") or 11436)
    # File GGUF tải riêng (không qua Ollama): tên model là tên file bỏ đuôi .gguf, tìm trong thư mục này và thư mục con.
    LOCAL_MODELS_DIR: Path = Path(os.environ.get("TOOL_DICH_MODELS_DIR") or (BASE_DIR / "tools" / "models"))
    DEFAULT_TRANSLATION_MODE: str = "NATURAL"
    DEFAULT_QA_LEVEL: str = "BALANCED"
    
    # Concurrency and batching
    DEFAULT_WORKER_COUNT: int = 1
    DEFAULT_CHUNK_TOKEN_TARGET: int = 750
    AUTOSAVE_DEBOUNCE_MS: int = 1000


settings = AppSettings()

# Ensure base directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
settings.FONTS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
