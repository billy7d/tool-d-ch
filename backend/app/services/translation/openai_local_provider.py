import json
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Any, Optional
from app.services.translation.provider_base import TranslationProvider
from app.services.translation.json_parser import parse_translation_batch_strict
from app.services.qa.result_validator import qa_error, validate_qa_result


class OpenAILocalProvider(TranslationProvider):
    """Provider for OpenAI-compatible local servers such as LM Studio, vLLM, or LocalAI."""

    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: str = "not-needed", default_model: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=Retry(total=2, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def health_check(self) -> bool:
        try:
            res = self.session.get(f"{self.base_url}/models", headers=self._headers(), timeout=2.0)
            return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            res = self.session.get(f"{self.base_url}/models", headers=self._headers(), timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                return [m.get("id") for m in data.get("data", [])]
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

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
        }

        try:
            res = self.session.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload, timeout=180.0)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"]
                parsed = parse_translation_batch_strict(content, expected_ids)
                if parsed.duplicate_ids or parsed.unknown_ids or parsed.raw_error:
                    raise RuntimeError(
                        f"Local AI batch mapping invalid: duplicate={parsed.duplicate_ids}, "
                        f"unknown={parsed.unknown_ids}, error={parsed.raw_error}"
                    )
                if parsed.translations:
                    return parsed.translations
                raise RuntimeError(f"Local AI returned unparsable JSON: {content[:200]}")
            else:
                raise RuntimeError(f"OpenAI Local error {res.status_code}: {res.text}")
        except Exception as e:
            raise RuntimeError(f"Local AI endpoint error: {e}")


    def translate_single(
        self,
        text: str,
        system_prompt: str,
        glossary_terms: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        user_prompt: Optional[str] = None,
    ) -> str:
        target_model = model or self.default_model
        glossary_hint = ""
        if glossary_terms:
            glossary_hint = "Thuật ngữ bắt buộc:\n" + "\n".join([f"- \"{k}\": \"{v}\"" for k, v in glossary_terms.items()]) + "\n\n"

        usr_msg = user_prompt or (
            f"{glossary_hint}"
            f"Hãy dịch chính xác đoạn văn bản sau sang tiếng Việt tự nhiên, chuẩn xác, đúng ngữ pháp:\n\n"
            f"{text}\n\n"
            f"YÊU CẦU BẮT BUỘC:\n"
            f"1. Văn xuôi phải bằng tiếng Việt; dịch đủ mọi ý, không tóm tắt hoặc bỏ câu.\n"
            f"2. Giữ nguyên tên riêng, nhãn hiệu, acronym, URL, code và identifier khi phù hợp.\n"
            f"3. Không đưa thêm ký tự CJK không có trong nguồn hoặc glossary bắt buộc.\n"
            f"4. Chỉ trả về nội dung bản dịch, không thêm lời chào, giải thích hay định dạng thừa."
        )


        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_msg}
            ],
            "temperature": temperature,
        }

        try:
            res = self.session.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload, timeout=90.0)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```") and content.endswith("```"):
                    content = re.sub(r'^```[a-zA-Z]*\n?', '', content)
                    content = re.sub(r'\n?```$', '', content).strip()
                if (content.startswith('"') and content.endswith('"')) or (content.startswith('“') and content.endswith('”')):
                    content = content[1:-1].strip()
                return content
            else:
                raise RuntimeError(f"OpenAI Local error {res.status_code}: {res.text}")
        except Exception as e:
            raise RuntimeError(f"Local AI single translation failed: {e}")

    def revise(self, source_text: str, current_translation: str, instruction: str, model: Optional[str] = None) -> str:
        target_model = model or self.default_model
        sys_msg = (
            "You are an expert Vietnamese editor and translator.\n"
            "Revise the Vietnamese translation based on the instruction.\n"
            "Output strictly 100% Vietnamese with NO Chinese characters (中文字符) or extraneous chatter."
        )
        usr_msg = f"SOURCE:\n{source_text}\n\nCURRENT:\n{current_translation}\n\nINSTRUCTION:\n{instruction}"
        payload = {
            "model": target_model,
            "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}],
            "temperature": 0.2,
        }
        try:
            res = self.session.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload, timeout=60.0)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```") and content.endswith("```"):
                    content = re.sub(r'^```[a-zA-Z]*\n?', '', content)
                    content = re.sub(r'\n?```$', '', content).strip()
                if (content.startswith('"') and content.endswith('"')) or (content.startswith('“') and content.endswith('”')):
                    content = content[1:-1].strip()
                return content
        except Exception as e:
            print(f"[OpenAILocalProvider.revise] Error: {e}")

        return current_translation



    def summarize_context(self, text_sample: str, model: Optional[str] = None, max_input_chars: Optional[int] = 4000) -> str:
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
        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Build structured source chapter memory. Return JSON only, without markdown or extra prose. "
                        "Follow this exact schema: " + json.dumps(schema, ensure_ascii=False)
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CHAPTER TITLE: {chapter_title}\nDOCUMENT TYPE: {document_type}\n"
                        f"SOURCE SAMPLE:\n{text_sample[:5500]}"
                    ),
                },
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(), json=payload, timeout=60.0,
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        return {}

    def extract_glossary(self, text_sample: str, document_type: str = "GENERAL", model: Optional[str] = None) -> List[Dict[str, str]]:
        return []

    def review_translation(self, source_text: str, translated_text: str, glossary_terms: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
        target_model = model or self.default_model
        system_message = (
            "Evaluate the English-to-Vietnamese translation for accuracy, omissions, "
            "hallucinations and locked glossary compliance. Return only JSON with "
            "is_passed, score, issues and suggested_revision."
        )
        user_message = (
            f"SOURCE:\n{source_text}\n\nTRANSLATION:\n{translated_text}\n\n"
            f"GLOSSARY:\n{json.dumps(glossary_terms, ensure_ascii=False)}"
        )
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=60.0,
            )
            if response.status_code != 200:
                return qa_error(f"OpenAI Local QA HTTP {response.status_code}: {response.text[:200]}")
            content = response.json()["choices"][0]["message"]["content"]
            return validate_qa_result(json.loads(content))
        except Exception as exc:
            return qa_error(f"OpenAI Local QA error: {exc}")

    def review_semantic_fidelity(self, source_text, translated_text, glossary_terms, entity_context, document_type, model=None):
        target_model = model or self.default_model
        system_message = (
            "You are a strict semantic fidelity critic for English-to-Vietnamese translation. "
            "Judge meaning only; accept natural non-literal Vietnamese restructuring. Do not rewrite or use outside knowledge. "
            "Return JSON only: status PASS|FAIL, score, errors, and checks for completeness, meaning, polarity, modality, causality, scope, entity_reference."
        )
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": json.dumps({
                    "source": source_text, "candidate": translated_text, "locked_glossary": glossary_terms,
                    "entities": entity_context, "document_type": document_type,
                }, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        response = self.session.post(
            f"{self.base_url}/chat/completions", headers=self._headers(), json=payload, timeout=60.0,
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI Local semantic critic HTTP {response.status_code}: {response.text[:200]}")
        return json.loads(response.json()["choices"][0]["message"]["content"])

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
            "You are a Vietnamese naturalness critic for an English-to-Vietnamese translation. Judge Vietnamese "
            "quality only and do not rewrite. Accept faithful non-literal Vietnamese. Check literal_calque, "
            "word_order, collocation, cohesion, pronoun_reference, register, redundancy, nominalization, "
            "passive_voice and sentence_flow. Do not penalize appropriate formal legal, technical or academic "
            "style. Return strict JSON only with status PASS or FAIL, score 0..1, all ten checks, and issues "
            "containing type, severity, target_span and message."
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
        }, ensure_ascii=False)
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        response = self.session.post(
            f"{self.base_url}/chat/completions", headers=self._headers(), json=payload, timeout=60.0,
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI Local naturalness critic HTTP {response.status_code}: {response.text[:200]}")
        return json.loads(response.json()["choices"][0]["message"]["content"])

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
            "You are a careful Vietnamese editorial translator. Edit the Vietnamese, not the meaning. Return "
            "only revised Vietnamese plain text. Keep every proposition, number, negation, modality, causality, "
            "condition, scope, entity, reference and locked term from the original source. Do not add, omit or "
            "reinterpret information. Only improve syntax, collocation, cohesion, pronouns and literal phrasing."
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
            "temperature": 0.1,
        }
        response = self.session.post(
            f"{self.base_url}/chat/completions", headers=self._headers(), json=payload, timeout=60.0,
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI Local editorial rewrite HTTP {response.status_code}: {response.text[:200]}")
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```") and content.endswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()
        return content
