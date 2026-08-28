from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.services.translation.model_capabilities import ModelCapabilities, get_model_capabilities


class TranslationProvider(ABC):
    def configure_context_window(self, context_window: int) -> None:
        """Khóa context provider theo ngân sách mà engine đã tính."""
        self._effective_context_window = int(context_window)

    def effective_context_window(self, model_name: str) -> int:
        configured = getattr(self, "_effective_context_window", None)
        if configured:
            return int(configured)
        capabilities = self.get_model_capabilities(model_name)
        return min(capabilities.context_window, capabilities.recommended_context_window)

    def get_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """Trả về ngân sách ngữ cảnh bảo thủ theo họ model."""
        return get_model_capabilities(model_name)

    @abstractmethod
    def health_check(self) -> bool:
        """Returns True if the LLM provider backend is available and responsive."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Returns a list of available model names."""
        pass

    @abstractmethod
    def translate(
        self,
        blocks: List[Dict[str, str]],  # [{"id": "node_1", "text": "..."}]
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3
    ) -> List[Dict[str, str]]:  # [{"node_id": "node_1", "text": "..."}]
        """Translates a batch of semantic text blocks into Vietnamese."""
        pass

    @abstractmethod
    def translate_single(
        self,
        text: str,
        system_prompt: str,
        glossary_terms: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        user_prompt: Optional[str] = None,
    ) -> str:
        """Translates a single text block directly into Vietnamese as a robust fallback."""
        pass

    @abstractmethod
    def revise(
        self,
        source_text: str,
        current_translation: str,
        instruction: str,
        model: Optional[str] = None
    ) -> str:
        """Re-translates or refines a specific text block with custom user instructions."""
        pass

    @abstractmethod
    def summarize_context(
        self,
        text_sample: str,
        model: Optional[str] = None,
        max_input_chars: Optional[int] = 4000,
    ) -> str:
        """Generates chapter memory or document topic summary."""
        pass

    @abstractmethod
    def build_chapter_memory(
        self,
        text_sample: str,
        chapter_title: str,
        document_type: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tạo Chapter Memory theo contract JSON có cấu trúc."""
        pass

    @abstractmethod
    def extract_glossary(
        self,
        text_sample: str,
        document_type: str = "GENERAL",
        model: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Extracts representative domain terms, named entities and suggested translations."""
        pass

    @abstractmethod
    def review_translation(
        self,
        source_text: str,
        translated_text: str,
        glossary_terms: Dict[str, str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Performs AI QA review checking accuracy, terminology, and naturalness."""
        pass

    def review_semantic_fidelity(
        self,
        source_text: str,
        translated_text: str,
        glossary_terms: Dict[str, str],
        entity_context: Dict[str, str],
        document_type: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Review semantic Phase 3; provider cũ có thể chưa hỗ trợ contract này."""
        raise NotImplementedError("Provider chưa hỗ trợ semantic-critic-v1")
