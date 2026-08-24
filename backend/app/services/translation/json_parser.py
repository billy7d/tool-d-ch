import re
import json
from typing import List, Dict, Any, Optional


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


def clean_and_parse_llm_json(content: str, expected_node_ids: Optional[List[str]] = None) -> List[Dict[str, str]]:
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
                    # Fallback to expected_node_ids by position if nid is missing or numeric
                    if (not nid or str(nid).isdigit()) and expected_node_ids and idx < len(expected_node_ids):
                        nid = expected_node_ids[idx]
                    if nid and txt:
                        results.append({"node_id": str(nid), "text": str(txt).strip()})
                elif isinstance(item, str) and expected_node_ids and idx < len(expected_node_ids):
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
