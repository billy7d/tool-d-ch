from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.services.translation.model_capabilities import ModelCapabilities, get_model_capabilities


PAIRWISE_NATURALNESS_SYSTEM_PROMPT = (
    "You compare two Vietnamese translations of the same English source for Vietnamese naturalness only. "
    "If one candidate adds, drops or changes information, prefer the other. Otherwise prefer the one a "
    "Vietnamese editor would publish as is. Return strict JSON only: "
    '{"preferred": "A" | "B" | "TIE", "reason": "<short reason>"}.'
)


_QUOTE_CHARS = "\"“”"


def unwrap_quoted_output(content: str, source_text: str) -> str:
    """Bỏ cặp ngoặc model bọc quanh cả bản dịch, nhưng không đụng tới lời thoại thật.

    Trước đây mọi output bắt đầu và kết thúc bằng ngoặc kép đều bị cắt, nên đoạn thoại kiểu
    `"Anh về rồi," cô nói. "Em đã chờ."` mất dấu mở và dấu đóng.
    """
    wrapped = (content.startswith('"') and content.endswith('"')) or (content.startswith("“") and content.endswith("”"))
    if not wrapped or len(content) < 2:
        return content
    source = (source_text or "").strip()
    inner = content[1:-1]
    if source[:1] in _QUOTE_CHARS or any(char in inner for char in _QUOTE_CHARS):
        return content
    return inner.strip()


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

    def review_naturalness(
        self,
        source_text: str,
        translated_text: str,
        document_type: str,
        register: str,
        sentence_style: str,
        previous_context: Any = None,
        glossary_terms: Optional[Dict[str, str]] = None,
        entity_context: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        style_hints: Optional[List[Dict[str, Any]]] = None,
        address_terms: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Đánh giá độ tự nhiên; provider cũ không được âm thầm coi là đạt."""
        raise NotImplementedError("Provider chưa hỗ trợ vietnamese naturalness critic")

    def editorial_rewrite(
        self,
        source_text: str,
        current_translation: str,
        naturalness_issues: List[Dict[str, Any]],
        document_type: str,
        register: str,
        sentence_style: str,
        glossary_terms: Optional[Dict[str, str]] = None,
        entity_context: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        previous_context: Any = None,
        address_terms: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Viết lại có kiểm soát; mặc định từ chối để tránh fallback không an toàn."""
        raise NotImplementedError("Provider chưa hỗ trợ editorial rewrite")

    def compare_naturalness(
        self,
        source_text: str,
        candidate_a: str,
        candidate_b: str,
        document_type: str = "GENERAL",
        register: str = "",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """So sánh cặp A/B; mô hình nhỏ chọn A/B đáng tin hơn chấm điểm tuyệt đối."""
        raise NotImplementedError("Provider chưa hỗ trợ pairwise naturalness")
