import re
from typing import List, Dict, Any, Optional
from app.db.models import NodeModel
from app.services.translation.vietnamese_post_processor import VietnamesePostProcessor


class DeterministicQA:
    NUMBER_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)?%?|\$\d+(?:[\.,]\d+)?\b")
    URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+")

    @classmethod
    def audit_node(cls, node: NodeModel) -> List[Dict[str, Any]]:
        issues = []
        source = (node.content or "").strip()
        translation = (node.translated_content or "").strip()

        if not translation:
            issues.append({
                "issue_type": "EMPTY_TRANSLATION",
                "severity": "ERROR",
                "message": "Bản dịch hiện đang để trống.",
                "source_snippet": source[:100],
                "translation_snippet": "",
                "suggested_fix": "Yêu cầu dịch lại đoạn này."
            })
            return issues

        # 1. Number Preservation Check (PRD Section 81)
        src_numbers = set(cls.NUMBER_PATTERN.findall(source))
        tr_numbers = set(cls.NUMBER_PATTERN.findall(translation))

        # Check for numbers in source that are missing in translation
        missing_numbers = src_numbers - tr_numbers
        really_missing = []
        for num in missing_numbers:
            clean_num = re.sub(r"[^\d]", "", num)
            if not any(clean_num in tr_n for tr_n in tr_numbers):
                really_missing.append(num)

        if really_missing:
            issues.append({
                "issue_type": "NUMBER_MISMATCH",
                "severity": "WARNING",
                "message": f"Thiếu số liệu hoặc tỷ lệ từ nguồn: {', '.join(really_missing)}",
                "source_snippet": source[:150],
                "translation_snippet": translation[:150],
                "suggested_fix": f"Bổ sung các con số ({', '.join(really_missing)}) vào bản dịch."
            })

        # 2. Length Anomaly / Possible Truncation Check (PRD Section 82)
        if len(source) > 100:
            ratio = len(translation) / len(source)
            if ratio < 0.25:
                issues.append({
                    "issue_type": "POSSIBLE_TRUNCATION",
                    "severity": "WARNING",
                    "message": f"Độ dài bản dịch quá ngắn so với bản gốc ({len(translation)} ký tự vs {len(source)} ký tự). Có thể đã bị cắt bớt nội dung.",
                    "source_snippet": source[:150],
                    "translation_snippet": translation[:150],
                    "suggested_fix": "Kiểm tra xem câu cuối hoặc các ý quan trọng có bị bỏ quên không."
                })
            elif ratio > 3.5 and len(source) > 80:
                issues.append({
                    "issue_type": "POSSIBLE_ADDED_CONTENT",
                    "severity": "WARNING",
                    "message": f"Độ dài bản dịch dài bất thường ({len(translation)} ký tự vs {len(source)} ký tự). Có thể mô hình AI đã tự giải thích thêm.",
                    "source_snippet": source[:150],
                    "translation_snippet": translation[:150],
                    "suggested_fix": "Cắt bỏ phần giải thích thừa hoặc diễn đạt gọn hơn."
                })

        # 3. URL and Link Preservation Check
        src_urls = set(cls.URL_PATTERN.findall(source))
        tr_urls = set(cls.URL_PATTERN.findall(translation))
        missing_urls = src_urls - tr_urls
        if missing_urls:
            issues.append({
                "issue_type": "BROKEN_REFERENCE",
                "severity": "WARNING",
                "message": f"Thiếu liên kết URL trong bản dịch: {', '.join(missing_urls)}",
                "source_snippet": source[:150],
                "translation_snippet": translation[:150],
                "suggested_fix": "Giữ nguyên các đường link URL gốc."
            })

        # 4. Chinese Character Detection (Strict Vietnamese Policy)
        if VietnamesePostProcessor.contains_chinese(translation):
            issues.append({
                "issue_type": "CONTAINS_CHINESE",
                "severity": "ERROR",
                "message": "Bản dịch bị dính chữ Hán / tiếng Trung Quốc.",
                "source_snippet": source[:150],
                "translation_snippet": translation[:150],
                "suggested_fix": "Bấm 'Dịch lại đoạn này với AI' để dịch sang 100% tiếng Việt sạch."
            })

        return issues
