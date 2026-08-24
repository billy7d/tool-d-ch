import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class TranslationParseResult:
    valid: bool
    translations: List[Dict[str, str]] = field(default_factory=list)
    missing_ids: List[str] = field(default_factory=list)
    duplicate_ids: List[str] = field(default_factory=list)
    unknown_ids: List[str] = field(default_factory=list)
    empty_ids: List[str] = field(default_factory=list)
    raw_error: Optional[str] = None


def extract_single_translation_text(raw_text: str) -> str:
    """
    Extracts clean translated text from a single node translation response.
    Handles:
    - JSON wrappers: {"translation": "..."}, {"text": "..."}, {"vietnamese": "..."}
    - Markdown code blocks: ```json ... ```
    - Smart quotes around JSON keys/values: “translation”: “...”
    - AI preambles and postambles (Lưu ý:..., Note:..., Ghi chú:...)
    """
    if not raw_text or not raw_text.strip():
        return ""

    cleaned = raw_text.strip()

    # 1. Strip markdown fences
    if "```" in cleaned:
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned).strip()

    # 2. Check if text contains JSON object (handle smart quotes “ ”)
    normalized_json_str = cleaned.replace('“', '"').replace('”', '"')
    bracket_match = re.search(r'(\{[\s\S]*\})', normalized_json_str)
    if bracket_match:
        candidate = bracket_match.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                for k in ["translation", "text", "translated_text", "vietnamese", "content", "result"]:
                    if k in parsed and isinstance(parsed[k], str):
                        cleaned = parsed[k]
                        break
        except Exception:
            # Regex extraction for "translation": "..."
            match = re.search(r'"(?:translation|text|vietnamese|translated_text|content)"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate)
            if match:
                cleaned = match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')

    # 3. Strip any AI postambles (Lưu ý:..., Ghi chú:..., Note:...)
    postamble_patterns = [
        r'\n*\s*(?:Lưu\s*[Ýý]|Ghi\s*chú|Chú\s*thích|Note|Notes)\s*[:\-].*$',
        r'\n*\s*-\s*Bản dịch đã được.*$',
        r'\n*\s*-\s*Đã loại bỏ.*$',
    ]
    for pattern in postamble_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    return cleaned.strip()


def clean_and_parse_llm_json(
    content: str,
    expected_node_ids: Optional[List[str]] = None,
    strict: bool = False,
) -> List[Dict[str, str]]:
    """
    Robustly parses LLM responses for translation batches.
    Handles:
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Preamble/postamble conversational text
    - Trailing commas in objects/arrays
    - Dict with 'translations' list
    - Direct list of translation objects
    - Dict of key -> value mapping
    - List of string translations matching expected_node_ids by order
    - Regex fallback for embedded json items
    """
    if not content or not content.strip():
        return []

    cleaned = content.strip()

    # 1. Strip markdown code fences if present
    if "```" in cleaned:
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned).strip()

    # 2. Try direct json.loads
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        # Try extracting the JSON block between outer brackets
        bracket_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', cleaned)
        if bracket_match:
            candidate = bracket_match.group(1)
            # Remove trailing commas: e.g. , } or , ]
            fixed = re.sub(r',\s*([\}\]])', r'\1', candidate)
            try:
                parsed = json.loads(fixed)
            except Exception:
                pass

    # 3. Interpret parsed data structures
    results: List[Dict[str, str]] = []
    if parsed is not None:
        raw_items = []
        if isinstance(parsed, dict):
            if "translations" in parsed and isinstance(parsed["translations"], list):
                raw_items = parsed["translations"]
            elif "results" in parsed and isinstance(parsed["results"], list):
                raw_items = parsed["results"]
            elif "nodes" in parsed and isinstance(parsed["nodes"], list):
                raw_items = parsed["nodes"]
            else:
                # Dict mapping node_id -> translation text
                for k, v in parsed.items():
                    if str(k).lower() not in ["model", "status", "version", "summary", "notes"]:
                        val_str = v.get("text") or v.get("translation") if isinstance(v, dict) else str(v)
                        results.append({"node_id": str(k), "text": str(val_str).strip()})
                if results:
                    return results
        elif isinstance(parsed, list):
            raw_items = parsed

        if raw_items:
            for idx, item in enumerate(raw_items):
                if isinstance(item, dict):
                    nid = item.get("node_id") or item.get("id")
                    txt = item.get("text") or item.get("translation") or item.get("translated_text") or item.get("vietnamese") or ""
                    # Chỉ chế độ tương thích cũ mới được phép ánh xạ theo vị trí.
                    if not strict and (not nid or str(nid).isdigit()) and expected_node_ids and idx < len(expected_node_ids):
                        nid = expected_node_ids[idx]
                    if nid:
                        results.append({"node_id": str(nid), "text": str(txt).strip()})
                elif not strict and isinstance(item, str) and expected_node_ids and idx < len(expected_node_ids):
                    results.append({"node_id": expected_node_ids[idx], "text": item.strip()})

            if results:
                return results

    # 4. Regex fallback: match {"node_id": "...", "text": "..."} patterns
    item_pattern = r'\{\s*"(?:node_id|id)"\s*:\s*"([^"]+)"\s*,\s*"(?:text|translation|translated_text|vietnamese)"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    matches = re.findall(item_pattern, content)
    if matches:
        for nid, txt in matches:
            clean_txt = txt.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').strip()
            results.append({"node_id": nid, "text": clean_txt})
        if results:
            return results

    # 5. Reverse key order Regex fallback: match {"text": "...", "node_id": "..."}
    reverse_pattern = r'\{\s*"(?:text|translation|translated_text|vietnamese)"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"(?:node_id|id)"\s*:\s*"([^"]+)"\s*\}'
    rev_matches = re.findall(reverse_pattern, content)
    if rev_matches:
        for txt, nid in rev_matches:
            clean_txt = txt.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').strip()
            results.append({"node_id": nid, "text": clean_txt})
        if results:
            return results

    return results


def validate_translation_batch(
    translations: List[Dict[str, str]],
    expected_node_ids: List[str],
    raw_error: Optional[str] = None,
) -> TranslationParseResult:
    """Đối chiếu ID tuyệt đối; không đoán node từ thứ tự đầu ra."""
    expected = list(dict.fromkeys(str(node_id) for node_id in expected_node_ids))
    expected_set = set(expected)
    seen: Dict[str, int] = {}
    normalized: List[Dict[str, str]] = []
    empty_ids: List[str] = []

    for item in translations or []:
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id") or item.get("id")
        if node_id is None:
            continue
        node_id = str(node_id)
        text = item.get("text")
        if text is None:
            text = item.get("translation") or item.get("translated_text") or ""
        text = str(text).strip()
        seen[node_id] = seen.get(node_id, 0) + 1
        normalized.append({"node_id": node_id, "text": text})
        if not text:
            empty_ids.append(node_id)

    duplicate_ids = sorted(node_id for node_id, count in seen.items() if count > 1)
    unknown_ids = sorted(node_id for node_id in seen if node_id not in expected_set)
    missing_ids = [node_id for node_id in expected if node_id not in seen]
    valid = not (missing_ids or duplicate_ids or unknown_ids or empty_ids or raw_error)
    return TranslationParseResult(
        valid=valid,
        translations=normalized,
        missing_ids=missing_ids,
        duplicate_ids=duplicate_ids,
        unknown_ids=unknown_ids,
        empty_ids=sorted(set(empty_ids)),
        raw_error=raw_error,
    )


def parse_translation_batch_strict(content: str, expected_node_ids: List[str]) -> TranslationParseResult:
    """Phân tích batch production và trả kết quả lỗi có cấu trúc."""
    if not content or not content.strip():
        return validate_translation_batch([], expected_node_ids, "MALFORMED_PROVIDER_RESPONSE")
    translations = clean_and_parse_llm_json(content, expected_node_ids, strict=True)
    error = None if translations else "MALFORMED_PROVIDER_RESPONSE"
    return validate_translation_batch(translations, expected_node_ids, error)
