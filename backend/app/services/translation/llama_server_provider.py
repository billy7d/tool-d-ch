"""Chạy đúng các prompt của OllamaProvider trên llama-server (llama.cpp).

Ollama không cho ghim slot cache và không nhận grammar. llama-server nhận cả hai, nên: lệnh có prompt riêng
(bộ nhớ chương, khôi phục chữ lạ) không còn xóa cache của lượt dịch, và model không thể viết bộ chữ không có
trong prompt (chữ Hán lẫn vào bản dịch là lý do chờ duyệt của 7/80 đoạn sách thật).

Mọi prompt, schema và tham số vẫn do OllamaProvider dựng; session ở đây chỉ đổi định dạng gửi đi và trả về.
"""
import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.services.translation.ollama_provider import OllamaProvider
from app.services.translation.provider_base import output_token_cap

# Mặc định của Ollama khi lệnh gọi không truyền; phải đặt tường minh vì mặc định của llama-server khác.
OLLAMA_DEFAULT_SAMPLING = {
    "temperature": 0.8, "top_k": 40, "top_p": 0.9, "min_p": 0.0, "typical_p": 1.0,
    "repeat_penalty": 1.1, "repeat_last_n": 64, "presence_penalty": 0.0, "frequency_penalty": 0.0,
}
SHARED_PREFIX_SLOT = 0
OTHER_PROMPT_SLOT = 1
# Bộ chữ qwen2.5 hay viết lẫn vào tiếng Việt: tên -> các dải mã Unicode (đầu, cuối).
FOREIGN_SCRIPTS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    # Bộ thủ, dấu câu và kana, chữ Hán mở rộng A, chữ Hán, chữ Hán tương thích, dạng toàn chiều rộng.
    "cjk": ((0x2E80, 0x2FDF), (0x3000, 0x312F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF)),
    "hangul": ((0x1100, 0x11FF), (0xAC00, 0xD7AF)),
    "cyrillic": ((0x0400, 0x052F),),
}
_SCRIPT_PATTERNS = {
    name: re.compile("[" + "".join(f"{chr(first)}-{chr(last)}" for first, last in ranges) + "]")
    for name, ranges in FOREIGN_SCRIPTS.items()
}
_GBNF_ESCAPE = chr(92) + "u"


def chatml_prompt(messages: List[Dict[str, str]]) -> str:
    """Đúng mẫu ChatML mà Ollama dùng cho qwen2.5, để model nhận cùng chuỗi token như khi chạy qua Ollama."""
    parts = [f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages]
    return "".join(parts) + "<|im_start|>assistant\n"


def script_ban_grammar(prompt_text: str) -> Optional[str]:
    """Grammar loại các bộ chữ không xuất hiện ở đâu trong prompt; nguồn hay glossary có thì vẫn được viết."""
    banned = [
        f"{_GBNF_ESCAPE}{first:04X}-{_GBNF_ESCAPE}{last:04X}"
        for name, ranges in FOREIGN_SCRIPTS.items() if not _SCRIPT_PATTERNS[name].search(prompt_text)
        for first, last in ranges
    ]
    if not banned:
        return None
    return "root ::= [^" + "".join(banned) + "]*"


def _request_key(payload: Dict[str, Any]) -> str:
    return json.dumps([payload.get("messages"), payload.get("options"), payload.get("format")], ensure_ascii=False, sort_keys=True)


class _OllamaShapedResponse:
    """Câu trả lời của llama-server /completion, đọc được như câu trả lời /api/chat của Ollama."""

    def __init__(self, response: requests.Response, slot: int):
        self.status_code = response.status_code
        self.text = response.text
        self._response = response
        self._slot = slot

    def json(self) -> Dict[str, Any]:
        body = self._response.json()
        timings = body.get("timings") or {}
        prompt_tokens = int(body.get("tokens_evaluated") or 0)
        return {
            "message": {"role": "assistant", "content": body.get("content", "")},
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": int(float(timings.get("prompt_ms") or 0) * 1e6),
            "eval_count": int(timings.get("predicted_n") or body.get("tokens_predicted") or 0),
            "eval_duration": int(float(timings.get("predicted_ms") or 0) * 1e6),
            "cached_tokens": int(timings.get("cache_n") if timings.get("cache_n") is not None
                                 else max(0, prompt_tokens - int(timings.get("prompt_n") or 0))),
            "slot": self._slot,
            "done": True,
        }


class _LlamaServerSession(requests.Session):
    def __init__(self, provider: "LlamaServerProvider"):
        super().__init__()
        self._provider = provider

    def post(self, url, data=None, json=None, **kwargs):
        if not isinstance(json, dict) or not str(url).endswith("/api/chat"):
            return super().post(url, data=data, json=json, **kwargs)
        provider = self._provider
        provider.ensure_server()
        key = _request_key(json)
        pending = provider._speculation
        if pending is not None and threading.current_thread() is pending["thread"]:
            body, slot = provider.completion_request(json)
            pending["response"] = _OllamaShapedResponse(super().post(f"{provider.base_url}/completion", json=body, **kwargs), slot)
            return pending["response"]
        busy = False
        if pending is not None:
            if pending["key"] == key:
                # Đúng yêu cầu đã dịch trước: chờ nó xong và dùng kết quả, không gọi model lần nữa.
                pending["thread"].join()
                provider._speculation = None
                if pending.get("response") is not None and pending["response"].status_code == 200:
                    provider.speculation_hits += 1
                    return pending["response"]
            else:
                busy = pending["thread"].is_alive()
        body, slot = provider.completion_request(json)
        if busy and provider.slots >= 2:
            # Slot dịch đang bận dịch trước đoạn kế tiếp; lệnh chấm/sửa của đoạn hiện tại chạy song song ở slot kia.
            slot = body["id_slot"] = OTHER_PROMPT_SLOT
        response = super().post(f"{provider.base_url}/completion", json=body, **kwargs)
        return _OllamaShapedResponse(response, slot)


class LlamaServerProvider(OllamaProvider):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11436",
        default_model: str = "qwen2.5:7b",
        ban_foreign_scripts: bool = True,
        slots: int = 2,
        manager: Any = None,
    ):
        super().__init__(base_url=manager.url if manager else base_url, default_model=default_model)
        self.ban_foreign_scripts = ban_foreign_scripts
        self.slots = slots
        self.manager = manager
        # File GGUF tải riêng (Gemma, Hy-MT...) không dùng ChatML: prompt ghép bằng mẫu hội thoại của chính model.
        self.use_model_template = bool(manager is not None and manager.uses_model_template(default_model))
        self.session = _LlamaServerSession(self)
        self._speculation: Optional[Dict[str, Any]] = None
        self.speculation_hits = 0
        self.speculation_starts = 0

    def speculate_translation(
        self, text: str, system_prompt: str, glossary_terms: Dict[str, str], model: str, temperature: float, user_prompt: str,
    ) -> bool:
        """Dịch trước một yêu cầu ở luồng phụ. Kết quả chỉ được dùng khi lệnh dịch thật sau đó trùng từng ký tự.

        Hai luồng sinh chữ cùng lúc trên GTX 1660 Super cho tổng tốc độ ~1,6 lần một luồng, nên đoạn kế tiếp được
        dịch trong lúc đoạn hiện tại còn đang chấm nghĩa. Đoán sai (đoạn hiện tại bị sửa hoặc chờ duyệt) thì prompt
        thật khác đi, kết quả dịch trước bị bỏ và model nhận đúng đầu vào như khi chạy tuần tự.
        """
        if self.slots < 2:
            return False
        previous = self._speculation
        if previous is not None and previous["thread"].is_alive():
            return False
        options = self._options(model, temperature, output_token_cap(text), top_p=0.85, repeat_penalty=self.VIETNAMESE_REPEAT_PENALTY)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        entry: Dict[str, Any] = {"key": _request_key({"messages": messages, "options": options}), "response": None}

        def run() -> None:
            try:
                self.translate_single(text, system_prompt, glossary_terms, model, temperature, user_prompt)
            except Exception:
                entry["response"] = None

        entry["thread"] = threading.Thread(target=run, name="translation-speculation", daemon=True)
        self._speculation = entry
        self.speculation_starts += 1
        entry["thread"].start()
        return True

    def ensure_server(self) -> None:
        """Có manager thì máy chủ được khởi động ở lệnh gọi đầu tiên, không phải lúc tạo provider."""
        if self.manager is None:
            return
        if not self.manager.ensure_running(self.default_model):
            raise RuntimeError("Không khởi động được llama-server.")
        self.slots = self.manager.slots or 1

    def health_check(self) -> bool:
        if self.manager is not None:
            # Chưa chạy nhưng khởi động được vẫn là sẵn sàng; việc nạp model do luồng dịch làm.
            return self.manager.available(self.default_model)
        try:
            return requests.get(f"{self.base_url}/health", timeout=2.0).status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        return [self.default_model] if self.health_check() else []

    def _options(self, model: str, temperature: float, num_predict: int, **extra: Any) -> Dict[str, Any]:
        options = super()._options(model, temperature, num_predict, **extra)
        # Số lớp trên GPU và số luồng do tiến trình llama-server quyết định khi khởi động.
        options.pop("num_gpu", None)
        options.pop("num_thread", None)
        return options

    def render_prompt(self, messages: List[Dict[str, str]]) -> str:
        if not self.use_model_template:
            return chatml_prompt(messages)
        response = requests.post(
            f"{self.base_url}/apply-template",
            json={"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}, timeout=30,
        )
        response.raise_for_status()
        return str(response.json()["prompt"])

    def completion_request(self, payload: Dict[str, Any]) -> tuple:
        messages = payload.get("messages") or []
        options = dict(payload.get("options") or {})
        sampling = {key: options.get(key, default) for key, default in OLLAMA_DEFAULT_SAMPLING.items()}
        prefix = getattr(self, "_prompt_prefix", "")
        system = next((message["content"] for message in messages if message["role"] == "system"), "")
        slot = SHARED_PREFIX_SLOT if self.slots < 2 or (prefix and system == prefix) else OTHER_PROMPT_SLOT
        body: Dict[str, Any] = {
            "prompt": self.render_prompt(messages),
            "n_predict": int(options.get("num_predict", -1)),
            "cache_prompt": True,
            "id_slot": slot,
            "stream": False,
            **sampling,
        }
        if "seed" in options:
            body["seed"] = options["seed"]
        response_format = payload.get("format")
        if response_format == "json":
            body["json_schema"] = {"type": "object"}
        elif isinstance(response_format, dict):
            body["json_schema"] = response_format
        elif self.ban_foreign_scripts:
            # Chỉ xét phần user (nguồn, glossary, ngữ cảnh đã duyệt). Phần system mang bộ nhớ chương do model viết
            # ở chế độ JSON, nơi không cấm được chữ lạ; một chữ Hán lọt vào đó không được làm tắt lệnh cấm cho cả chương.
            grammar = script_ban_grammar("\n".join(
                str(message.get("content", "")) for message in messages if message.get("role") != "system"
            ))
            if grammar:
                body["grammar"] = grammar
        return body, slot


__all__ = ["LlamaServerProvider", "chatml_prompt", "script_ban_grammar", "OLLAMA_DEFAULT_SAMPLING"]
