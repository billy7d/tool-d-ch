from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelCapabilities:
    context_window: int = 4096
    recommended_context_window: int = 4096
    recommended_source_tokens: int = 1200
    supports_json_mode: bool = True
    supports_system_prompt: bool = True


# Bảng này chỉ chứa họ model; logic ngữ cảnh không phụ thuộc một model cụ thể.
MODEL_FAMILY_CAPABILITIES: Dict[str, ModelCapabilities] = {
    "qwen": ModelCapabilities(8192, 4096, 1200, True, True),
    "llama": ModelCapabilities(8192, 4096, 1200, True, True),
    "gemma": ModelCapabilities(8192, 4096, 1100, True, True),
    "mistral": ModelCapabilities(8192, 4096, 1200, True, True),
    "mock": ModelCapabilities(4096, 4096, 1200, True, True),
}


def get_model_capabilities(model_name: str) -> ModelCapabilities:
    normalized = (model_name or "").lower()
    for family, capabilities in MODEL_FAMILY_CAPABILITIES.items():
        if family in normalized:
            return capabilities
    return ModelCapabilities()
