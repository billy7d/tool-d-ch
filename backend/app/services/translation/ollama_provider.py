import json
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.translation.provider_base import TranslationProvider
from app.services.translation.json_parser import parse_translation_batch_strict
from app.services.qa.result_validator import qa_error, validate_qa_result


class OllamaProvider(TranslationProvider):
    def __init__(self, base_url: str = settings.DEFAULT_OLLAMA_URL, default_model: str = settings.DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        
        # High-performance persistent HTTP session with connection pooling
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=Retry(total=2, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def health_check(self) -> bool:
        try:
            res = self.session.get(f"{self.base_url}/api/tags", timeout=2.0)
            return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            res = self.session.get(f"{self.base_url}/api/tags", timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                return [m.get("name") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def translate(
        self,
        blocks: List[Dict[str, str]],
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2
    ) -> List[Dict[str, str]]:
        target_model = model or self.default_model
        expected_ids = [b.get("id") or b.get("node_id", "") for b in blocks if b.get("id") or b.get("node_id")]
        capabilities = self.get_model_capabilities(target_model)

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "60m",  # Keep model locked in VRAM for instant subsequent responses
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": self.effective_context_window(target_model),
            }
        }

        try:
            res = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=180.0)
            if res.status_code == 200:
                resp_json = res.json()
                content = resp_json.get("message", {}).get("content", "")
                parsed = parse_translation_batch_strict(content, expected_ids)
                if parsed.duplicate_ids or parsed.unknown_ids or parsed.raw_error:
                    raise RuntimeError(
                        f"Ollama batch mapping invalid: duplicate={parsed.duplicate_ids}, "
                        f"unknown={parsed.unknown_ids}, error={parsed.raw_error}"
                    )
                if parsed.translations:
                    return parsed.translations
                raise RuntimeError(f"Ollama returned unparsable content: {content[:200]}")
            else:
                raise RuntimeError(f"Ollama error {res.status_code}: {res.text}")
        except Exception as e:
            raise RuntimeError(f"Ollama translation failed: {e}")

    def translate_single(
        self,
        text: str,
        system_prompt: str,
        glossary_terms: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.15,
        user_prompt: Optional[str] = None,
    ) -> str:
        """
        Direct single-node translation with ultra-high reliability.
        Does not rely on complex multi-item JSON batches.
        """
        target_model = model or self.default_model
        capabilities = self.get_model_capabilities(target_model)
        
        glossary_hint = ""
        if glossary_terms:
            glossary_hint = "Thuật ngữ bắt buộc:\n" + "\n".join([f"- \"{k}\": \"{v}\"" for k, v in glossary_terms.items()]) + "\n\n"

        usr_msg = user_prompt or (
            f"{glossary_hint}"
            f"Hãy dịch chính xác đoạn văn bản tiếng Anh sau sang tiếng Việt xuất bản tự nhiên, hoàn chỉnh:\n\n"
            f"{text}\n\n"
            f"YÊU CẦU BẮT BUỘC:\n"
            f"1. Chỉ trả về VĂN BẢN THUẦN (Plain text), TUYỆT ĐỐI KHÔNG dùng JSON, KHÔNG bọc vào {{...}}.\n"
            f"2. TUYỆT ĐỐI KHÔNG thêm bất kỳ ghi chú, phần 'Lưu ý', lời giải thích hay lời chào nào.\n"
            f"3. Dịch ĐẦY ĐỦ 100% tất cả các câu từ đầu đến cuối, không được ngắt quãng hay bỏ lửng.\n"
            f"4. Văn xuôi phải bằng tiếng Việt; được giữ nguyên tên riêng, nhãn hiệu, acronym, URL, code và identifier khi phù hợp.\n"
            f"5. Không đưa thêm ký tự CJK không có trong nguồn hoặc glossary bắt buộc."
        )

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_msg}
            ],
            "stream": False,
            "keep_alive": "60m",
            "options": {
                "temperature": temperature,
                "top_p": 0.85,
                "repeat_penalty": 1.25,
                "num_ctx": self.effective_context_window(target_model),
            }
        }

        try:
            res = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=90.0)
            if res.status_code == 200:
                resp_json = res.json()
                content = resp_json.get("message", {}).get("content", "").strip()
                # Clean markdown backticks if returned
                if content.startswith("```") and content.endswith("```"):
                    content = re.sub(r'^```[a-zA-Z]*\n?', '', content)
                    content = re.sub(r'\n?```$', '', content).strip()
                # Clean quotation wrappers if entire text was quoted
                if (content.startswith('"') and content.endswith('"')) or (content.startswith('“') and content.endswith('”')):
                    content = content[1:-1].strip()
                return content
            else:
                raise RuntimeError(f"Ollama error {res.status_code}: {res.text}")
        except Exception as e:
            raise RuntimeError(f"Ollama single translation failed: {e}")

    def revise(
        self,
        source_text: str,
        current_translation: str,
        instruction: str,
        model: Optional[str] = None
    ) -> str:
        target_model = model or self.default_model
        capabilities = self.get_model_capabilities(target_model)
        sys_msg = (
            "You are an expert Vietnamese book editor and translator.\n"
            "Produce a refined, publishing-quality Vietnamese translation of the given English source text according to the user instruction.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY plain Vietnamese text (NO JSON, NO notes, NO 'Lưu ý', NO markdown quotes).\n"
            "2. 100% complete Vietnamese without repeating sentences.\n"
            "3. Absolutely NO Chinese characters (中文字符), pinyin, or CJK punctuation (，。；)."
        )

        # If current_translation contains Chinese or is empty, ignore it to prevent model contagion
        if not current_translation or bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]', current_translation)):
            usr_msg = (
                f"ENGLISH SOURCE:\n{source_text}\n\n"
                f"USER INSTRUCTION:\n{instruction}\n\n"
                f"VIETNAMESE TRANSLATION (Plain text only):"
            )
        else:
            usr_msg = (
                f"ENGLISH SOURCE:\n{source_text}\n\n"
                f"DRAFT TRANSLATION:\n{current_translation}\n\n"
                f"USER INSTRUCTION:\n{instruction}\n\n"
                f"REVISED VIETNAMESE TRANSLATION (Plain text only):"
            )

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}
            ],
            "stream": False,
            "keep_alive": "60m",
            "options": {
                "temperature": 0.15,
                "top_p": 0.85,
                "repeat_penalty": 1.25,
                "num_ctx": self.effective_context_window(target_model),
            }
        }

        try:
            res = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
            if res.status_code == 200:
                content = res.json().get("message", {}).get("content", "").strip()
                if content.startswith("```") and content.endswith("```"):
                    content = re.sub(r'^```[a-zA-Z]*\n?', '', content)
                    content = re.sub(r'\n?```$', '', content).strip()
                if (content.startswith('"') and content.endswith('"')) or (content.startswith('“') and content.endswith('”')):
                    content = content[1:-1].strip()
                return content
            else:
                print(f"[OllamaProvider.revise] Error {res.status_code}: {res.text}")
        except Exception as e:
            print(f"[OllamaProvider.revise] Request failed: {e}")

        return current_translation


    def summarize_context(self, text_sample: str, model: Optional[str] = None, max_input_chars: Optional[int] = 4000) -> str:
        target_model = model or self.default_model
        sys_msg = "Summarize the key topic, main characters/entities, terminology, and tone of this section in 3-4 bullet points (in Vietnamese)."
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": text_sample[:max_input_chars] if max_input_chars else text_sample}
            ],
            "stream": False,
            "keep_alive": "60m",
            "options": {"temperature": 0.2}
        }
        try:
            res = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=40.0)
            if res.status_code == 200:
                return res.json().get("message", {}).get("content", "").strip()
        except Exception:
            pass
        return ""

    def build_chapter_memory(
        self,
        text_sample: str,
        chapter_title: str,
        document_type: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        target_model = model or self.default_model
        schema = {
            "summary": "...", "entities": [{"source": "...", "preferred": "..."}],
            "key_concepts": [], "tone": "...", "pronoun_notes": [],
            "terminology": [{"source": "...", "preferred": "..."}],
            "important_facts": [], "style_notes": [],
        }
        system_message = (
            "Build structured source chapter memory for an English-to-Vietnamese translation engine. "
            "Return JSON only, without markdown or prose outside JSON. Follow this exact schema: "
            + json.dumps(schema, ensure_ascii=False)
        )
        user_message = (
            f"CHAPTER TITLE: {chapter_title}\nDOCUMENT TYPE: {document_type}\n"
            f"SOURCE SAMPLE:\n{text_sample[:5500]}"
        )
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "60m",
            "options": {"temperature": 0.1},
        }
        try:
            response = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "")
                parsed = json.loads(content)
                return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        return {}

    def extract_glossary(
        self,
        text_sample: str,
        document_type: str = "GENERAL",
        model: Optional[str] = None
    ) -> List[Dict[str, str]]:
        target_model = model or self.default_model
        sys_msg = (
            f"You are a professional lexicographer specializing in {document_type} translations. "
            "Extract 10-25 key specialist terms, proper names, and domain concepts from the English text, and provide natural, standard Vietnamese translations. "
            "Return output as a JSON object with schema: {\"terms\": [{\"source_term\": \"...\", \"target_term\": \"...\", \"category\": \"...\", \"notes\": \"...\"}]}"
        )
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": text_sample[:6000]}
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "60m",
            "options": {"temperature": 0.2}
        }
        try:
            res = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
            if res.status_code == 200:
                content = res.json().get("message", {}).get("content", "")
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "terms" in parsed:
                    return parsed["terms"]
        except Exception:
            pass
        return []

    def review_translation(
        self,
        source_text: str,
        translated_text: str,
        glossary_terms: Dict[str, str],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        target_model = model or self.default_model
        sys_msg = (
            "Evaluate this English to Vietnamese translation. Check for meaning accuracy, missing facts, hallucinations, Vietnamese naturalness, and glossary compliance. "
            "Return JSON: {\"is_passed\": bool, \"score\": float, \"issues\": [\"...\"], \"suggested_revision\": \"...\"}"
        )
        usr_msg = (
            f"SOURCE: {source_text}\n"
            f"TRANSLATION: {translated_text}\n"
            f"GLOSSARY: {json.dumps(glossary_terms, ensure_ascii=False)}"
        )
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "60m",
            "options": {"temperature": 0.1}
        }
        try:
            res = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
            if res.status_code == 200:
                content = res.json().get("message", {}).get("content", "")
                return validate_qa_result(json.loads(content))
            return qa_error(f"Ollama QA HTTP {res.status_code}: {res.text[:200]}")
        except Exception as exc:
            return qa_error(f"Ollama QA error: {exc}")

    def review_semantic_fidelity(self, source_text, translated_text, glossary_terms, entity_context, document_type, model=None):
        target_model = model or self.default_model
        system_message = (
            "You are a strict semantic fidelity critic for English-to-Vietnamese translation. "
            "Judge meaning only, not literal wording or harmless Vietnamese restructuring. Do not rewrite. "
            "Use no outside knowledge. Return JSON only with status PASS or FAIL, score 0..1, errors array, "
            "and checks for completeness, meaning, polarity, modality, causality, scope, entity_reference. "
            "Allowed error types: SEMANTIC_OMISSION, SEMANTIC_ADDITION, MEANING_DRIFT, MODALITY_ERROR, "
            "CAUSALITY_ERROR, SCOPE_ERROR, CONDITION_ERROR, COMPARISON_ERROR, ENTITY_REFERENCE_ERROR, PRONOUN_AMBIGUITY."
        )
        user_message = json.dumps({
            "source": source_text, "candidate": translated_text,
            "locked_glossary": glossary_terms, "entities": entity_context,
            "document_type": document_type,
        }, ensure_ascii=False)
        payload = {
            "model": target_model,
            "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
            "format": "json", "stream": False, "keep_alive": "60m",
            "options": {"temperature": 0.0, "num_ctx": self.effective_context_window(target_model)},
        }
        response = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama semantic critic HTTP {response.status_code}: {response.text[:200]}")
        return json.loads(response.json().get("message", {}).get("content", ""))

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
        target_model = model or self.default_model
        system_message = (
            "You are a Vietnamese naturalness critic for an English-to-Vietnamese translation. "
            "Judge Vietnamese quality only. Do not rewrite the sentence and do not penalize a faithful "
            "non-literal translation. Ask whether the Vietnamese would read as natural writing by a Vietnamese "
            "author in the stated domain when the English source is hidden. Check literal calques, English word "
            "order, collocation, cohesion, pronoun/reference, register, redundancy, nominalization, unnecessary "
            "passive voice and sentence flow. Formal legal, technical and academic style is acceptable when it is "
            "appropriate and semantically precise. Return strict JSON only with status PASS or FAIL, score 0..1, "
            "the ten required checks, and issues. Each issue must contain type, severity, target_span and message."
        )
        user_message = json.dumps({
            "source": source_text,
            "vietnamese_translation": translated_text,
            "document_type": document_type,
            "register": register,
            "sentence_style": sentence_style,
            "previous_bilingual_context": previous_context or [],
            "relevant_glossary": glossary_terms or {},
            "relevant_entities": entity_context or {},
            "required_check_names": [
                "literal_calque", "word_order", "collocation", "cohesion", "pronoun_reference",
                "register", "redundancy", "nominalization", "passive_voice", "sentence_flow",
            ],
        }, ensure_ascii=False)
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "60m",
            "options": {"temperature": 0.0, "num_ctx": self.effective_context_window(target_model)},
        }
        response = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama naturalness critic HTTP {response.status_code}: {response.text[:200]}")
        return json.loads(response.json().get("message", {}).get("content", ""))

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
        target_model = model or self.default_model
        system_message = (
            "You are a careful Vietnamese editorial translator. Edit the Vietnamese, not the meaning. "
            "Return only the revised Vietnamese plain text, with no JSON, notes or markdown. Keep every proposition "
            "from ORIGINAL SOURCE. Do not add or omit information, change numbers, negation, modality, causality, "
            "scope, conditions, entities, references or locked terminology. Do not invent a new interpretation. "
            "Only improve Vietnamese syntax, collocation, cohesion, pronouns and unnecessary literal phrasing. "
            "Respect the document domain, register and sentence style."
        )
        user_message = json.dumps({
            "ORIGINAL_SOURCE": source_text,
            "CURRENT_VIETNAMESE_TRANSLATION": current_translation,
            "NATURALNESS_ISSUES": naturalness_issues,
            "DOCUMENT_TYPE": document_type,
            "REGISTER": register,
            "SENTENCE_STYLE": sentence_style,
            "LOCKED_GLOSSARY": glossary_terms or {},
            "RELEVANT_ENTITIES": entity_context or {},
            "INSTRUCTION": "Edit the Vietnamese, not the meaning.",
        }, ensure_ascii=False)
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "keep_alive": "60m",
            "options": {"temperature": 0.1, "num_ctx": self.effective_context_window(target_model)},
        }
        response = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=60.0)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama editorial rewrite HTTP {response.status_code}: {response.text[:200]}")
        content = response.json().get("message", {}).get("content", "").strip()
        if content.startswith("```") and content.endswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()
        return content
