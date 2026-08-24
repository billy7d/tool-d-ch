from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.services.translation.prompt_profiles import get_style_pack


@dataclass(frozen=True)
class TranslationConfig:
    source_language: str
    target_language: str
    model_name: str
    translation_mode: str
    document_type: str
    register: str
    sentence_style: str
    custom_instructions: str
    style_guide: Dict[str, Any] = field(default_factory=dict)
    quality_profile: str = "BALANCED"

    @classmethod
    def from_project(
        cls,
        project: Any,
        model_override: Optional[str] = None,
        custom_instruction_override: Optional[str] = None,
        translation_mode_override: Optional[str] = None,
        document_type_override: Optional[str] = None,
        register_override: Optional[str] = None,
        sentence_style_override: Optional[str] = None,
    ) -> "TranslationConfig":
        style_guide = dict(getattr(project, "style_guide", None) or {})
        document_type = str(document_type_override or getattr(project, "document_type", None) or "GENERAL").upper()
        translation_mode = str(translation_mode_override or getattr(project, "translation_mode", None) or "NATURAL").upper()
        domain_default = get_style_pack(document_type)
        default_register = domain_default.register.upper()
        if document_type in {"LEGAL", "TECHNICAL", "ACADEMIC"}:
            default_sentence_style = "PRESERVE"
        else:
            default_sentence_style = "MODERATE"
        register = str(register_override or style_guide.get("register") or default_register).upper()
        sentence_style = str(sentence_style_override or style_guide.get("sentence_style") or default_sentence_style).upper()
        normalized_style = {
            key: value for key, value in style_guide.items()
            if key not in {"register", "sentence_style"}
        }
        normalized_style.update({"register": register, "sentence_style": sentence_style})
        custom_value = custom_instruction_override
        if custom_value is None:
            custom_value = getattr(project, "custom_instructions", None)
        return cls(
            source_language=str(getattr(project, "source_language", None) or "en"),
            target_language=str(getattr(project, "target_language", None) or "vi"),
            model_name=str(model_override or getattr(project, "selected_model", None) or "qwen2.5:7b"),
            translation_mode=translation_mode,
            document_type=document_type,
            register=register,
            sentence_style=sentence_style,
            custom_instructions=str(custom_value or "").strip(),
            style_guide=normalized_style,
            quality_profile=str(getattr(project, "qa_level", None) or "BALANCED").upper(),
        )

    def signature_payload(self) -> Dict[str, Any]:
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "model_name": self.model_name,
            "translation_mode": self.translation_mode,
            "document_type": self.document_type,
            "register": self.register,
            "sentence_style": self.sentence_style,
            "custom_instructions": self.custom_instructions,
            "style_guide": self.style_guide,
            "quality_profile": self.quality_profile,
        }
