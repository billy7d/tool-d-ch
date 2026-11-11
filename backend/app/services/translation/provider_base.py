from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.services.translation.model_capabilities import ModelCapabilities, get_model_capabilities


PAIRWISE_NATURALNESS_SYSTEM_PROMPT = (
    "You compare two Vietnamese translations of the same English source for Vietnamese naturalness only. "
    "If one candidate adds, drops or changes information, prefer the other. Otherwise prefer the one a "
    "Vietnamese editor would publish as is. Return strict JSON only: "
    '{"preferred": "A" | "B" | "TIE", "reason": "<short reason>"}.'
)


# Đo trên sách thật: critic 7B hay viết lời giải thích dài (200–290 token, ~15s) và báo "thêm ý" cho
# cụm có sẵn trong nguồn. Trích đúng cụm tiếng Anh trong nháy đơn để engine tự bác được lời sai.
SEMANTIC_CRITIC_BREVITY = (
    " Report only real meaning errors. Before reporting an addition, check the source: a phrase that "
    "appears in the source is not an addition. Each error message must be at most 20 words and quote the "
    "exact English phrase from the source in single quotes. Write error messages in English."
)
# Không mô tả hình dạng câu trả lời PASS: đo thật, hễ prompt nhắc "return status PASS with an empty
# errors array" (kể cả kèm "include all seven checks") là qwen2.5:7b chép đúng hình dạng tối giản đó,
# bỏ checks, và luật "PASS phải báo đủ checks" chặn 9/20 đoạn sách.

_VERDICT = {"type": "string", "enum": ["PASS", "FAIL"]}
_SEMANTIC_CHECKS = ["completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference"]
# Ép định dạng để model không thể bỏ checks: chạy lại 32 lượt critic thật, 3 lượt ERROR vì thiếu checks
# đều thành kết luận hợp lệ, số FAIL giữ nguyên 16. "status" phải đứng trước "errors": đặt errors lên đầu
# thì qwen2.5:7b luôn nghĩ ra lỗi (32/32 FAIL). maxItems chặn lặp một lỗi tới hết num_predict.
SEMANTIC_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "status": _VERDICT,
        "score": {"type": "number"},
        "checks": {
            "type": "object",
            "properties": {name: _VERDICT for name in _SEMANTIC_CHECKS},
            "required": _SEMANTIC_CHECKS,
        },
        "errors": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": [
                        "SEMANTIC_OMISSION", "SEMANTIC_ADDITION", "MEANING_DRIFT", "MODALITY_ERROR", "CAUSALITY_ERROR",
                        "SCOPE_ERROR", "CONDITION_ERROR", "COMPARISON_ERROR", "ENTITY_REFERENCE_ERROR", "PRONOUN_AMBIGUITY",
                    ]},
                    "message": {"type": "string"},
                },
                "required": ["type", "message"],
            },
        },
    },
    "required": ["status", "score", "checks", "errors"],
}

_TERM_LIST = {
    "type": "object",
    "properties": {"source": {"type": "string"}, "preferred": {"type": "string"}},
    "required": ["source", "preferred"],
}
_NOTE_LIST = {"type": "array", "maxItems": 3, "items": {"type": "string"}}
# Không có schema, model trả pronoun_notes dạng object và cả Chapter Memory bị loại ở mọi lần chạy thật.
# Giới hạn số mục vì bộ nhớ này nằm trong mọi prompt: mỗi token lặp lại làm mọi token sau chậm đi.
CHAPTER_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entities": {"type": "array", "maxItems": 8, "items": _TERM_LIST},
        "key_concepts": _NOTE_LIST,
        "tone": {"type": "string"},
        "pronoun_notes": _NOTE_LIST,
        "terminology": {"type": "array", "maxItems": 8, "items": _TERM_LIST},
        "important_facts": _NOTE_LIST,
        "style_notes": _NOTE_LIST,
    },
    "required": [
        "summary", "entities", "key_concepts", "tone", "pronoun_notes", "terminology", "important_facts", "style_notes",
    ],
}

REVIEW_TASK_PREAMBLE = (
    "REVIEW TASK. Do not translate anything now; the rules above describe the translator whose work you are checking. "
)
EDIT_TASK_PREAMBLE = "EDITING TASK. For this request the instructions below replace the output contract above. "
_CHARS_PER_TOKEN_CONSERVATIVE = 3  # Thực đo ~4 ký tự/token với prompt Anh-Việt.


def output_token_cap(*texts: str, factor: float = 2.0, floor: int = 256, ceiling: int = 2048) -> int:
    """Trần số token output theo độ dài đầu vào (~4 ký tự/token, tiếng Việt tốn token hơn tiếng Anh)."""
    source_tokens = sum(len(text or "") for text in texts) / 4
    return int(min(ceiling, max(floor, source_tokens * factor + 64)))


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

    def remember_prompt_prefix(self, prefix: str) -> None:
        """Engine báo phần đầu prompt ổn định của lượt dịch gần nhất để lệnh kiểm tra dùng chung."""
        self._prompt_prefix = prefix or ""

    def review_messages(
        self,
        system_message: str,
        user_message: str,
        max_output_tokens: int,
        model_name: str,
        preamble: str = REVIEW_TASK_PREAMBLE,
    ) -> List[Dict[str, str]]:
        """Dựng message cho lệnh kiểm tra/biên tập sao cho không xóa cache prompt của lượt dịch.

        Ollama chỉ giữ một prompt cho mỗi slot và mặc định chỉ có một slot, nên một system prompt khác làm
        lượt dịch sau phải đọc lại ~1.500 token (~7 giây). Chạy lại 32 lượt critic thật với phần đầu dùng
        chung: 9 lượt đổi kết luận, đọc từng ca thì 7 lượt đúng hơn và 2 lượt kém đi. Prompt quá dài thì
        dùng lại dạng cũ.
        """
        prefix = getattr(self, "_prompt_prefix", "")
        if prefix:
            merged = f"{preamble}{system_message}\n\nINPUT:\n{user_message}"
            estimated = (len(prefix) + len(merged)) / _CHARS_PER_TOKEN_CONSERVATIVE + max_output_tokens
            if estimated <= self.effective_context_window(model_name):
                return [{"role": "system", "content": prefix}, {"role": "user", "content": merged}]
        return [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

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
        raise NotImplementedError("Provider chưa hỗ trợ semantic critic")

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
