import re
from typing import List, Dict, Any


class ParagraphRepair:
    HYPHENATED_COMPOUNDS = {
        "state-of-the-art", "user-friendly", "up-to-date", "real-time",
        "long-term", "short-term", "cost-effective", "well-known",
        "high-level", "low-level", "decision-making", "cross-cutting",
        "peer-to-peer", "end-to-end", "follow-up", "built-in",
        "out-of-date", "all-out", "step-by-step", "part-time", "full-time"
    }

    @classmethod
    def repair_hyphenation(cls, text: str) -> str:
        """
        Repairs line-wrap hyphens:
        'invest-\nment' -> 'investment'
        'state-of-\nthe-art' -> 'state-of-the-art'
        """
        # First check known compound multi-word patterns
        for comp in cls.HYPHENATED_COMPOUNDS:
            parts = comp.split("-")
            for i in range(1, len(parts)):
                prefix = "-".join(parts[:i])
                suffix = "-".join(parts[i:])
                broken_pattern = re.compile(rf"\b{re.escape(prefix)}-\s*\n?\s*{re.escape(suffix)}\b", re.IGNORECASE)
                text = broken_pattern.sub(comp, text)

        def replace_hyphen(match):
            word1 = match.group(1)
            word2 = match.group(2)
            
            # If word2 starts with uppercase (e.g. Austro-Hungarian), keep hyphen
            if word2 and word2[0].isupper():
                return f"{word1}-{word2}"

            # Common prepositions / articles that should keep hyphen in compound
            if word1.lower() in ["of", "by", "to", "in", "on", "and", "co", "pre", "post", "re"] and len(word1) <= 3:
                # If it looks like part of hyphenated phrase
                return f"{word1}{word2}"
                
            # Default to joining the broken word
            return f"{word1}{word2}"

        # Match word ending with hyphen followed by newline or space then word
        pattern = r"([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)"
        repaired = re.sub(pattern, replace_hyphen, text)
        return repaired

    @classmethod
    def join_broken_lines(cls, lines: List[str]) -> str:
        if not lines:
            return ""

        joined_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if not joined_lines:
                joined_lines.append(line_str)
                continue

            prev = joined_lines[-1]
            if prev.endswith("-") and not prev.endswith(" -"):
                word1 = prev.rstrip("-")
                joined_lines[-1] = cls.repair_hyphenation(f"{word1}-\n{line_str}")
            else:
                joined_lines.append(f" {line_str}")

        result = "".join(joined_lines)
        result = re.sub(r"[ \t]+", " ", result).strip()
        return result

    @classmethod
    def sort_reading_order(cls, blocks: List[Dict[str, Any]], page_width: float = 600.0) -> List[Dict[str, Any]]:
        if len(blocks) <= 1:
            return blocks

        col_threshold = page_width / 2.0
        left_col = []
        right_col = []
        full_width = []

        left_count = 0
        right_count = 0

        for b in blocks:
            bbox = b.get("bbox", [0, 0, 0, 0])
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
            width = x1 - x0
            
            if width > (page_width * 0.7):
                full_width.append(b)
            elif x1 <= col_threshold + 20:
                left_col.append(b)
                left_count += 1
            elif x0 >= col_threshold - 20:
                right_col.append(b)
                right_count += 1
            else:
                full_width.append(b)

        if left_count >= 2 and right_count >= 2:
            left_col.sort(key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
            right_col.sort(key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
            full_width.sort(key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
            return sorted(full_width, key=lambda b: b.get("bbox", [0, 0, 0, 0])[1]) + left_col + right_col

        return sorted(blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))
