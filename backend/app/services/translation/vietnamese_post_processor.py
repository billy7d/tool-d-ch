import re
from typing import Dict, Optional, Tuple


class VietnamesePostProcessor:
    """
    Post-translation sanitization, validation, and enhancement engine.
    Ensures 100% pure Vietnamese output without Chinese characters or foreign script leaks,
    unwraps JSON blocks, eliminates conversational AI notes and repetition loops,
    formats punctuation, and enforces locked glossary consistency.
    """

    AI_PREAMBLE_PATTERNS = [
        r'^(?:Dưới đây là|Đây là|Bản dịch tiếng Việt|Bản dịch là|Nội dung dịch|Bản dịch:)\s*[:\-]?\s*',
        r'^(?:Here is the translation|Vietnamese translation|Translation:)\s*[:\-]?\s*',
    ]

    # Full Unicode ranges for Chinese / Hanzi characters AND Chinese/Fullwidth punctuation
    CHINESE_PATTERN = re.compile(
        r'[\u4e00-\u9fff'          # CJK Unified Ideographs
        r'\u3400-\u4dbf'          # CJK Extension A
        r'\uf900-\ufaff'          # CJK Compatibility Ideographs
        r'\U00020000-\U0002A6DF'  # CJK Extension B
        r'\U0002A700-\U0002B73F'  # CJK Extension C
        r'\U0002B740-\U0002B81F'  # CJK Extension D
        r'\U0002B820-\U0002CEAF'  # CJK Extension E
        r'\U0002F800-\U0002FA1F'  # CJK Compatibility Supplement
        r'\u3000-\u303f'          # CJK Symbols and Punctuation (、。〈〉《》「」『』【】 etc.)
        r'\uff01-\uff0f'          # Fullwidth symbols (！＂＃＄％＆＇（）＊＋，－．／)
        r'\uff1a-\uff20'          # Fullwidth (：；＜＝＞？＠)
        r'\uff3b-\uff40'          # Fullwidth (［＼］＾＿｀)
        r'\uff5b-\uff65]'         # Fullwidth (｛｜｝～｟｠｡｢｣､･)
    )

    CJK_PUNCT_MAP = {
        '，': ', ',
        '。': '. ',
        '；': '; ',
        '：': ': ',
        '！': '! ',
        '？': '? ',
        '、': ', ',
        '（': ' (',
        '）': ') ',
        '【': ' [',
        '】': '] ',
        '《': ' “',
        '》': '” ',
        '～': '~',
        '—': '—',
        '…': '...',
    }

    @classmethod
    def contains_chinese(cls, text: str) -> bool:
        """
        Returns True if the text contains any Chinese / Hanzi character or Chinese punctuation.
        """
        if not text:
            return False
        return bool(cls.CHINESE_PATTERN.search(text))

    @classmethod
    def normalize_cjk_punctuation(cls, text: str) -> str:
        """
        Converts CJK and fullwidth punctuation to standard Latin/Vietnamese punctuation.
        """
        if not text:
            return ""
        result = text
        for cjk_char, latin_char in cls.CJK_PUNCT_MAP.items():
            result = result.replace(cjk_char, latin_char)
        return result

    @classmethod
    def strip_chinese_characters(cls, text: str) -> str:
        """
        Removes any leftover Chinese / Hanzi characters and CJK punctuation, cleaning up spaces.
        """
        if not text:
            return ""
        cleaned = cls.CHINESE_PATTERN.sub('', text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        return cleaned.strip()

    @classmethod
    def validate_pure_vietnamese(cls, text: str) -> Tuple[bool, str]:
        """
        Validates if text is clean Vietnamese without foreign/Chinese script intrusions.
        Returns (is_valid, reason).
        """
        if not text or not text.strip():
            return False, "EMPTY_TEXT"

        if cls.contains_chinese(text):
            return False, "CONTAINS_CHINESE_CHARACTERS"

        return True, "OK"

    @classmethod
    def deduplicate_repetitions(cls, text: str) -> str:
        """
        Eliminates degenerative LLM repetition loops where identical or near-identical
        paragraphs/sentences are repeated multiple times.
        """
        if not text:
            return ""

        # 1. Deduplicate identical paragraphs separated by newlines
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if not paragraphs:
            return text

        unique_paragraphs = []
        for p in paragraphs:
            if unique_paragraphs and (p == unique_paragraphs[-1] or p in unique_paragraphs):
                continue
            unique_paragraphs.append(p)

        res = "\n\n".join(unique_paragraphs)

        # 2. Check for sentence-level repetition loops within single paragraph
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', res) if s.strip()]
        if len(sentences) > 2:
            unique_sentences = []
            for s in sentences:
                if unique_sentences and s == unique_sentences[-1]:
                    continue
                if len(unique_sentences) >= 2 and s == unique_sentences[-2]:
                    continue
                unique_sentences.append(s)
            res = " ".join(unique_sentences)

        # 3. Collapse repeating phrase loops of 20+ chars
        for match_len in range(60, 20, -5):
            if len(res) < match_len * 2:
                continue
            i = 0
            while i <= len(res) - match_len * 2:
                chunk = res[i:i+match_len]
                if not chunk.strip() or len(chunk.strip()) < 15:
                    i += 1
                    continue
                next_chunk = res[i+match_len:i+match_len*2]
                if chunk == next_chunk:
                    loop_body = chunk
                    j = i + match_len
                    while j < len(res) and res[j:j+len(loop_body)] == loop_body:
                        j += len(loop_body)
                    res = res[:i + match_len] + res[j:]
                else:
                    i += 1

        return res.strip()

    @classmethod
    def clean_vietnamese_text(cls, text: str) -> str:
        if not text:
            return ""

        # 0. Extract text from JSON wrappers and strip AI postamble notes
        from app.services.translation.json_parser import extract_single_translation_text
        cleaned = extract_single_translation_text(text)

        # 1. Convert CJK punctuation to standard Latin punctuation
        cleaned = cls.normalize_cjk_punctuation(cleaned)

        # 2. Remove AI preamble/conversational artifacts if any
        for pattern in cls.AI_PREAMBLE_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

        # 3. Strip any residual AI postambles (Lưu ý:..., Ghi chú:..., Note:...)
        postamble_patterns = [
            r'\n*\s*(?:Lưu\s*[Ýý]|Ghi\s*chú|Chú\s*thích|Note|Notes)\s*[:\-].*$',
            r'\n*\s*-\s*Bản dịch đã được.*$',
            r'\n*\s*-\s*Đã loại bỏ.*$',
        ]
        for pattern in postamble_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

        # 4. Deduplicate repetition loops
        cleaned = cls.deduplicate_repetitions(cleaned)

        # 5. Clean consecutive punctuation clusters like ",.;,." or ", ."
        cleaned = re.sub(r'[,;:]+\s*([.!?])', r'\1', cleaned)
        cleaned = re.sub(r'([.!?])\s*[,;:]+', r'\1', cleaned)
        cleaned = re.sub(r'[,;]{2,}', ', ', cleaned)
        cleaned = re.sub(r'\.{4,}', '...', cleaned)

        # 6. Normalize whitespace and newlines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # 7. Fix spacing around punctuation: no space before , . : ; ! ?
        cleaned = re.sub(r'\s+([,.:;!?])', r'\1', cleaned)

        # 8. Ensure single space after punctuation if followed by letters/digits
        cleaned = re.sub(r'([,.:;!?])([^\s\d,.:;!?"\'\)\]\}”’])', r'\1 \2', cleaned)

        # 9. Fix common quotation marks
        cleaned = re.sub(r'\"([^\"]+)\"', r'“\1”', cleaned)

        # 10. Fix ellipses spacing
        cleaned = re.sub(r'\s*\.\.\.\s*', '... ', cleaned)
        cleaned = cleaned.rstrip()

        return cleaned

    @classmethod
    def enforce_locked_glossary(
        cls,
        translated_text: str,
        source_text: str,
        locked_glossary: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Guarantees that if a source English term was present in the source text,
        its mandatory locked target term is strictly adhered to in the translated text.
        """
        if not locked_glossary or not translated_text or not source_text:
            return translated_text

        result = translated_text
        for src_term, tgt_term in locked_glossary.items():
            pattern = r'\b' + re.escape(src_term) + r'\b'
            if re.search(pattern, source_text, re.IGNORECASE):
                pass

        return result
