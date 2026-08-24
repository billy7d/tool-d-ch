import json
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Any, Optional
from app.services.translation.provider_base import TranslationProvider
from app.services.translation.json_parser import clean_and_parse_llm_json


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
                parsed_translations = clean_and_parse_llm_json(content, expected_node_ids=expected_ids)
                if parsed_translations:
                    return parsed_translations
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
        temperature: float = 0.2
    ) -> str:
        target_model = model or self.default_model
        glossary_hint = ""
        if glossary_terms:
            glossary_hint = "Thuật ngữ bắt buộc:\n" + "\n".join([f"- \"{k}\": \"{v}\"" for k, v in glossary_terms.items()]) + "\n\n"

        usr_msg = (
            f"{glossary_hint}"
            f"Hãy dịch chính xác đoạn văn bản sau sang tiếng Việt tự nhiên, chuẩn xác, đúng ngữ pháp:\n\n"
            f"{text}\n\n"
            f"YÊU CẦU BẮT BUỘC:\n"
            f"1. Bản dịch phải là 100% TIẾNG VIỆT HOÀN CHỈNH.\n"
            f"2. Tuyệt đối KHÔNG ĐƯỢC chứa chữ Hán / tiếng Trung Quốc (中文字符) hay bất kỳ chữ Hán nào.\n"
            f"3. Chỉ trả về duy nhất nội dung bản dịch tiếng Việt, không thêm lời chào, giải thích hay định dạng thừa."
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



    def summarize_context(self, text_sample: str, model: Optional[str] = None) -> str:
        return ""

    def extract_glossary(self, text_sample: str, document_type: str = "GENERAL", model: Optional[str] = None) -> List[Dict[str, str]]:
        return []

    def review_translation(self, source_text: str, translated_text: str, glossary_terms: Dict[str, str], model: Optional[str] = None) -> Dict[str, Any]:
        return {"is_passed": True, "score": 1.0, "issues": [], "suggested_revision": ""}
