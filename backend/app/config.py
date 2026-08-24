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
    DATA_DIR: Path = BASE_DIR / "data"
    PROJECTS_DIR: Path = BASE_DIR / "data" / "projects"
    FONTS_DIR: Path = BASE_DIR / "fonts"
    LOGS_DIR: Path = BASE_DIR / "logs"
    STATIC_DIR: Path = BASE_DIR / "frontend" / "dist"
    
    # AI Engine defaults
    DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "qwen2.5:7b"
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
