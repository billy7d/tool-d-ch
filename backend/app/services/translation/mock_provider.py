from typing import List, Dict, Any, Optional
from app.services.translation.provider_base import TranslationProvider
from app.services.translation.term_matcher import TermMatcher


class MockProvider(TranslationProvider):
    """
    High-fidelity Mock translation provider for unit tests, offline development,
    and stress testing 1,000 pages without demanding live GPU resources.
    """

    SAMPLE_DICTIONARY = {
        "introduction": "Giới thiệu",
        "chapter": "Chương",
        "the market continued to expand": "Thị trường tiếp tục tăng trưởng",
        "investment": "đầu tư",
        "cash flow": "dòng tiền",
        "neural network": "mạng nơ-ron",
        "opportunity cost": "chi phí cơ hội",
        "compound interest": "lãi kép",
    }

    def health_check(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return ["mock-qwen2.5:7b", "mock-llama3.1:8b", "mock-gemma2:9b"]

    def translate(
        self,
        blocks: List[Dict[str, str]],
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3
    ) -> List[Dict[str, str]]:
        translations = []
        for block in blocks:
            node_id = block.get("id") or block.get("node_id", "")
            text = block.get("text", "")
            
            # Simple mock translation heuristic
            vi_text = f"[Bản dịch tiếng Việt] {text}"
            for en, vi in self.SAMPLE_DICTIONARY.items():
                if en in text.lower():
                    vi_text = vi_text.replace(en, vi)
                    
            translations.append({
                "node_id": node_id,
                "text": vi_text
            })
        return translations

    def translate_single(
        self,
        text: str,
        system_prompt: str,
        glossary_terms: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        user_prompt: Optional[str] = None,
    ) -> str:
        vi_text = f"[Bản dịch tiếng Việt] {text}"
        for en, vi in self.SAMPLE_DICTIONARY.items():
            if en in text.lower():
                vi_text = vi_text.replace(en, vi)
        if glossary_terms:
            for en_term, vi_term in glossary_terms.items():
                if TermMatcher.contains(text, en_term):
                    vi_text = TermMatcher.pattern(en_term).sub(vi_term, vi_text)
        return vi_text


    def revise(
        self,
        source_text: str,
        current_translation: str,
        instruction: str,
        model: Optional[str] = None
    ) -> str:
        return f"{current_translation} (Đã chỉnh sửa theo yêu cầu: {instruction})"

    def summarize_context(self, text_sample: str, model: Optional[str] = None, max_input_chars: Optional[int] = 4000) -> str:
        return "- Chủ đề: Tài chính và Đầu tư\n- Thuật ngữ chính: Dòng tiền, Lãi kép\n- Văn phong: Tự nhiên, rõ ràng"

    def build_chapter_memory(
        self,
        text_sample: str,
        chapter_title: str,
        document_type: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "summary": f"Chương '{chapter_title}' thuộc lĩnh vực {document_type}.",
            "entities": [],
            "key_concepts": [],
            "tone": "Rõ ràng, nhất quán.",
            "pronoun_notes": [],
            "terminology": [],
            "important_facts": [],
            "style_notes": [],
        }

    def extract_glossary(
        self,
        text_sample: str,
        document_type: str = "GENERAL",
        model: Optional[str] = None
    ) -> List[Dict[str, str]]:
        return [
            {"source_term": "cash flow", "target_term": "dòng tiền", "category": "FINANCE", "notes": "Thuật ngữ chuẩn"},
            {"source_term": "compound interest", "target_term": "lãi kép", "category": "FINANCE", "notes": "Thuật ngữ chuẩn"},
            {"source_term": "opportunity cost", "target_term": "chi phí cơ hội", "category": "FINANCE", "notes": "Kinh tế học"},
            {"source_term": "neural network", "target_term": "mạng nơ-ron", "category": "TECHNICAL", "notes": "Trí tuệ nhân tạo"},
        ]

    def review_translation(
        self,
        source_text: str,
        translated_text: str,
        glossary_terms: Dict[str, str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "status": "PASS",
            "is_passed": True,
            "score": 0.98,
            "issues": [],
            "suggested_revision": "",
            "error": None,
        }
