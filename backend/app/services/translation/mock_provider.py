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

    def review_semantic_fidelity(
        self,
        source_text: str,
        translated_text: str,
        glossary_terms: Dict[str, str],
        entity_context: Dict[str, str],
        document_type: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mock có tín hiệu cố định để CI không cần Ollama."""
        marker_map = {
            "[[OMISSION]]": "SEMANTIC_OMISSION",
            "[[ADDITION]]": "SEMANTIC_ADDITION",
            "[[DRIFT]]": "MEANING_DRIFT",
            "[[MODALITY]]": "MODALITY_ERROR",
            "[[CAUSALITY]]": "CAUSALITY_ERROR",
            "[[SCOPE]]": "SCOPE_ERROR",
            "[[CONDITION]]": "CONDITION_ERROR",
            "[[COMPARISON]]": "COMPARISON_ERROR",
            "[[ENTITY]]": "ENTITY_REFERENCE_ERROR",
            "[[PRONOUN]]": "PRONOUN_AMBIGUITY",
        }
        issue_type = next((value for marker, value in marker_map.items() if marker in translated_text), None)
        if not issue_type:
            return {
                "status": "PASS", "score": 0.98, "errors": [],
                "checks": {key: "PASS" for key in ("completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference")},
            }
        check = {
            "MODALITY_ERROR": "modality", "CAUSALITY_ERROR": "causality", "SCOPE_ERROR": "scope",
            "ENTITY_REFERENCE_ERROR": "entity_reference", "SEMANTIC_OMISSION": "completeness",
        }.get(issue_type, "meaning")
        checks = {key: "PASS" for key in ("completeness", "meaning", "polarity", "modality", "causality", "scope", "entity_reference")}
        checks[check] = "FAIL"
        return {
            "status": "FAIL", "score": 0.35,
            "errors": [{"type": issue_type, "severity": "ERROR", "source_span": "", "target_span": "", "message": "Lỗi semantic được gieo trong fixture."}],
            "checks": checks,
        }

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
    ) -> Dict[str, Any]:
        """Mock critic có các tín hiệu ổn định để test ranking và editorial flow offline."""
        checks = {key: "PASS" for key in (
            "literal_calque", "word_order", "collocation", "cohesion", "pronoun_reference",
            "register", "redundancy", "nominalization", "passive_voice", "sentence_flow",
        )}
        target = translated_text or ""
        lowered = target.lower()
        issues = []
        if "[[NATURALNESS_ERROR]]" in target:
            raise RuntimeError("mock naturalness critic failure")
        if "[[NATURALNESS_LOW]]" in target:
            checks["sentence_flow"] = "FAIL"
            issues.append({
                "type": "SENTENCE_FLOW", "severity": "WARNING", "target_span": "[[NATURALNESS_LOW]]",
                "message": "Câu vẫn khó đọc trong ngữ cảnh tiếng Việt.",
            })
            return {"status": "FAIL", "score": 0.42, "checks": checks, "issues": issues}

        literal_patterns = {
            "đưa ra quyết định": ("collocation", "COLLOCATION", "Cụm từ này là collocation dịch sát, dư thừa trong tiếng Việt."),
            "một quyết định": ("collocation", "COLLOCATION", "Cụm danh từ này có thể diễn đạt bằng động từ tự nhiên hơn."),
            "thực hiện hành động": ("literal_calque", "LITERAL_CALQUE", "Cách ghép động từ còn mang dấu vết dịch từng từ."),
            "các bước để": ("word_order", "WORD_ORDER", "Trật tự và cách chọn danh từ còn bám cấu trúc tiếng Anh."),
            "tiến hành việc": ("redundancy", "REDUNDANCY", "Cụm từ rườm rà, không cần thiết trong câu này."),
            "được phê duyệt bởi": ("passive_voice", "PASSIVE_VOICE", "Bị động theo trật tự tiếng Anh chưa tự nhiên trong văn cảnh này."),
            "được công bố bởi": ("passive_voice", "PASSIVE_VOICE", "Bị động theo trật tự tiếng Anh chưa tự nhiên trong văn cảnh này."),
            "được ký bởi": ("passive_voice", "PASSIVE_VOICE", "Bị động theo trật tự tiếng Anh chưa tự nhiên trong văn cảnh này."),
            "ngay lập tức": ("redundancy", "REDUNDANCY", "Có thể dùng trạng từ ngắn gọn hơn trong văn phong tự nhiên."),
            "là dễ dàng để": ("word_order", "WORD_ORDER", "Cấu trúc tính từ còn theo trật tự tiếng Anh."),
            "để có thể": ("redundancy", "REDUNDANCY", "Cụm mở đầu có thể rút gọn mà không mất nghĩa."),
            "với sự tôn trọng đối với": ("literal_calque", "LITERAL_CALQUE", "Cụm giới từ bị dịch từng từ."),
            "khác biệt từ những gì": ("collocation", "COLLOCATION", "Collocation này chưa tự nhiên trong tiếng Việt."),
            "theo một cách": ("redundancy", "REDUNDANCY", "Cụm trạng ngữ có thể diễn đạt gọn hơn."),
            "các sự ": ("nominalization", "NOMINALIZATION", "Danh từ hóa làm câu nặng và không tự nhiên."),
            "được xây dựng thông qua": ("passive_voice", "PASSIVE_VOICE", "Cấu trúc bị động và giới từ còn mang dấu vết nguồn."),
            "đang được tập trung vào việc": ("passive_voice", "PASSIVE_VOICE", "Cấu trúc bị động và danh từ hóa không cần thiết."),
            "đã thực hiện việc tăng": ("nominalization", "NOMINALIZATION", "Danh từ hóa và động từ đệm làm câu nặng."),
            "đã được giảm bởi": ("passive_voice", "PASSIVE_VOICE", "Có thể chuyển sang chủ động tự nhiên hơn."),
            "được cải thiện khi các chi phí": ("passive_voice", "PASSIVE_VOICE", "Cấu trúc bị động liên tiếp làm câu nặng."),
            "một tổng quan của": ("collocation", "COLLOCATION", "Collocation này chưa phù hợp với tiếng Việt tự nhiên."),
            "để xem xét lại": ("word_order", "WORD_ORDER", "Cách kết hợp động từ chưa tự nhiên."),
            "làm một sự khác biệt": ("collocation", "COLLOCATION", "Cụm từ bị dịch sát từng thành phần."),
            "thời hạn cuối cùng": ("redundancy", "REDUNDANCY", "Danh từ ghép bị kéo dài không cần thiết."),
            "sau một sự xem xét": ("nominalization", "NOMINALIZATION", "Danh từ hóa có thể chuyển thành mệnh đề động từ."),
            "dẫn đến trong": ("literal_calque", "LITERAL_CALQUE", "Cụm động từ chưa có collocation tiếng Việt tự nhiên."),
            "một lợi nhuận": ("collocation", "COLLOCATION", "Danh từ không cần mạo từ dịch thành một trong câu này."),
            "ở tại": ("word_order", "WORD_ORDER", "Giới từ bị dịch sát, không tự nhiên trong ngữ cảnh."),
            "mặc dù sự": ("cohesion", "COHESION", "Cách danh từ hóa làm quan hệ câu nặng nề."),
            "được chịu sự phụ thuộc": ("passive_voice", "PASSIVE_VOICE", "Cấu trúc bị động và danh từ hóa không cần thiết."),
            "với rủi ro quá mức": ("collocation", "COLLOCATION", "Cụm từ chưa phải collocation tự nhiên."),
            "một biên độ": ("collocation", "COLLOCATION", "Thuật ngữ tài chính không nên dịch theo cấu trúc mạo từ này."),
            "một sự nghỉ ngơi": ("nominalization", "NOMINALIZATION", "Có thể dùng động từ trực tiếp thay cho danh từ hóa."),
            "là bình thường để": ("word_order", "WORD_ORDER", "Cấu trúc vị ngữ còn theo tiếng Anh."),
            "không phải để giải quyết": ("word_order", "WORD_ORDER", "Cấu trúc động từ còn theo tiếng Anh."),
            "có thể dường như": ("redundancy", "REDUNDANCY", "Hai lớp mức độ làm câu rườm rà."),
            "tạo ra một thói quen": ("collocation", "COLLOCATION", "Động từ và danh từ chưa kết hợp tự nhiên."),
            "làm cho nó dễ dàng hơn để": ("word_order", "WORD_ORDER", "Cấu trúc gây khiến còn bám tiếng Anh."),
            "sự nghi ngờ xuất hiện": ("cohesion", "COHESION", "Cách danh từ hóa chưa tự nhiên trong câu mở đầu."),
            "là không hợp lệ": ("word_order", "WORD_ORDER", "Phủ định nên đặt trực tiếp trước tính từ."),
            "là bị thiếu": ("word_order", "WORD_ORDER", "Cấu trúc bị động không cần thiết."),
            "thực hiện một yêu cầu": ("collocation", "COLLOCATION", "Động từ đệm làm cụm kỹ thuật nặng."),
            "trước khi nó": ("pronoun_reference", "PRONOUN_REFERENCE", "Đại từ lặp không cần thiết trong tiếng Việt."),
            "các cuộc gọi mạng lặp lại": ("collocation", "COLLOCATION", "Cụm danh từ dài và bám trật tự tiếng Anh."),
            "một sự giảm": ("nominalization", "NOMINALIZATION", "Danh từ hóa không cần thiết."),
            "bằng cách sử dụng": ("redundancy", "REDUNDANCY", "Cụm phương thức có thể rút gọn."),
            "được giải thích bởi": ("passive_voice", "PASSIVE_VOICE", "Bị động theo tiếng Anh chưa phù hợp ngữ cảnh."),
            "lời giải thích được đề xuất": ("passive_voice", "PASSIVE_VOICE", "Cụm bị động có thể diễn đạt tự nhiên hơn."),
            "được đưa vào trong tài khoản": ("literal_calque", "LITERAL_CALQUE", "Thành ngữ bị dịch từng từ."),
            "tối ưu hóa của": ("word_order", "WORD_ORDER", "Cấu trúc sở hữu chưa tự nhiên."),
            "theo sau chỉ nếu": ("word_order", "WORD_ORDER", "Trật tự mệnh đề điều kiện còn bám nguồn."),
            "được thay đổi sau": ("passive_voice", "PASSIVE_VOICE", "Bị động không cần thiết trong mệnh đề này."),
            "đồng ý với nó": ("pronoun_reference", "PRONOUN_REFERENCE", "Đại từ thay thế không tự nhiên trong tiếng Việt."),
            "chỉ nếu thông báo bằng văn bản được cung cấp": ("word_order", "WORD_ORDER", "Mệnh đề điều kiện còn giữ trật tự tiếng Anh."),
            "đã nói rằng cô ấy": ("pronoun_reference", "PRONOUN_REFERENCE", "Đại từ lặp làm câu văn nặng."),
            "hầu như ở trên": ("literal_calque", "LITERAL_CALQUE", "Cụm miêu tả bị dịch sát thành ngữ."),
            "được bao quanh bởi sự": ("passive_voice", "PASSIVE_VOICE", "Bị động và danh từ hóa làm câu thiếu tự nhiên."),
        }
        for phrase, (check, issue_type, message) in literal_patterns.items():
            if phrase in lowered:
                checks[check] = "FAIL"
                issues.append({
                    "type": issue_type, "severity": "WARNING", "target_span": phrase, "message": message,
                })
        if "[[NATURALNESS_FAIL]]" in target and not issues:
            checks["literal_calque"] = "FAIL"
            issues.append({
                "type": "LITERAL_CALQUE", "severity": "WARNING", "target_span": "[[NATURALNESS_FAIL]]",
                "message": "Mock fixture đánh dấu candidate dịch sát.",
            })
        if issues:
            return {"status": "FAIL", "score": 0.58, "checks": checks, "issues": issues}
        return {"status": "PASS", "score": 0.96, "checks": checks, "issues": []}

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
    ) -> str:
        """Sửa một số fixture đại diện; các candidate khác được giữ nguyên để test fail-closed."""
        if "[[EDITORIAL_REWRITE_FAIL]]" in current_translation:
            return current_translation
        source = (source_text or "").strip().lower()
        target = current_translation or ""
        if "decided to take action immediately" in source:
            return "Công ty quyết định hành động ngay."
        if "proposal was approved by the board" in source:
            return "Hội đồng quản trị đã phê duyệt đề xuất."
        if "đưa ra quyết định" in target.lower():
            return target.replace("đã đưa ra quyết định để thực hiện hành động ngay lập tức", "quyết định hành động ngay")
        return target
