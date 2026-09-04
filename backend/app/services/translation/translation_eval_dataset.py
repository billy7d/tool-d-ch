"""Bộ dữ liệu eval P1/P2 và schema nhãn human.

Corpus này ghép các fixture P2/P3/P0 đã có với 24 mẫu adversarial bổ sung để
đạt 220 mẫu. Điểm human để trống cho tới khi có blind A/B review; benchmark
offline không được tự nhận là đánh giá tự nhiên như người bản ngữ.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


EVAL_DATASET_VERSION = "p1-p2-vietnamese-quality-eval-v1"
DOMAINS = (
    "GENERAL", "BUSINESS", "FINANCE", "SELF_HELP",
    "TECHNICAL", "ACADEMIC", "LEGAL", "LITERATURE",
)
SCORE_FIELDS = (
    "semantic_accuracy", "naturalness", "terminology",
    "contextual_coherence", "register",
)
ADVERSARIAL_TAGS = {
    "negation", "double_negation", "modality", "condition", "causality",
    "percent", "date", "currency", "number", "entity", "pronoun", "long_paragraph",
    "idiom", "collocation", "business_collocation", "phrasal_verb", "passive", "nominalization", "polysemy",
    "multi_paragraph", "technical_definition", "legal_obligation", "academic_hedging",
    "discourse_connector", "sentence_merge", "comparison", "personification", "sensory_detail",
}
HUMAN_EVAL_COLUMNS = (
    "sample_id", "domain", "source", "baseline_translation", "candidate_translation",
    "semantic_accuracy", "naturalness", "terminology", "contextual_coherence",
    "register", "preferred_output", "notes",
)


_EXTRA_SAMPLES: List[Dict[str, Any]] = [
    {"sample_id": "extra-general-01", "domain": "GENERAL", "source": "The plan was not rejected because the evidence was incomplete.", "preferred_output": "Kế hoạch không bị bác bỏ vì bằng chứng chưa đầy đủ.", "adversarial_tags": ["negation", "causality", "passive"]},
    {"sample_id": "extra-general-02", "domain": "GENERAL", "source": "No one said that the second warning was unnecessary.", "preferred_output": "Không ai nói rằng cảnh báo thứ hai là không cần thiết.", "adversarial_tags": ["double_negation", "number"]},
    {"sample_id": "extra-general-03", "domain": "GENERAL", "source": "Alice sent the revised file to Bob after the committee had reviewed the schedule, the budget, and the sequence of dependent tasks, but he could not open it because the archive was damaged and no replacement copy had been prepared.", "preferred_output": "Alice gửi tệp đã chỉnh sửa cho Bob sau khi ủy ban xem xét lịch trình, ngân sách và thứ tự các tác vụ phụ thuộc, nhưng anh ấy không thể mở vì kho lưu trữ đã hỏng và chưa có bản sao thay thế.", "adversarial_tags": ["pronoun", "modality", "entity", "long_paragraph", "causality"]},
    {"sample_id": "extra-business-01", "domain": "BUSINESS", "source": "The board may delay the launch if demand does not recover by June.", "preferred_output": "Hội đồng quản trị có thể trì hoãn việc ra mắt nếu nhu cầu không phục hồi trước tháng Sáu.", "adversarial_tags": ["modality", "condition", "date", "business_collocation"]},
    {"sample_id": "extra-business-02", "domain": "BUSINESS", "source": "The partnership created $10 million in revenue, not profit.", "preferred_output": "Quan hệ hợp tác tạo ra 10 triệu đô la doanh thu, không phải lợi nhuận.", "adversarial_tags": ["currency", "negation", "entity"]},
    {"sample_id": "extra-business-03", "domain": "BUSINESS", "source": "The new process reduces handoffs and makes follow-up easier.", "preferred_output": "Quy trình mới giảm số lần bàn giao và giúp việc liên hệ lại dễ dàng hơn.", "adversarial_tags": ["collocation", "phrasal_verb"]},
    {"sample_id": "extra-finance-01", "domain": "FINANCE", "source": "The interest on the loan is not the same as investor interest in the company.", "preferred_output": "Tiền lãi của khoản vay không giống với sự quan tâm của nhà đầu tư đối với công ty.", "adversarial_tags": ["polysemy", "negation"]},
    {"sample_id": "extra-finance-02", "domain": "FINANCE", "source": "The return was 4.25% in Q3, while the return of the device was delayed.", "preferred_output": "Mức sinh lời trong quý III là 4,25%, trong khi việc trả lại thiết bị bị trì hoãn.", "adversarial_tags": ["polysemy", "percent", "passive"]},
    {"sample_id": "extra-finance-03", "domain": "FINANCE", "source": "A wider margin does not necessarily mean lower exposure to risk.", "preferred_output": "Biên rộng hơn không nhất thiết có nghĩa là mức độ phơi nhiễm với rủi ro thấp hơn.", "adversarial_tags": ["polysemy", "negation", "academic_hedging"]},
    {"sample_id": "extra-self-help-01", "domain": "SELF_HELP", "source": "You do not need to avoid every difficult feeling to move forward.", "preferred_output": "Bạn không cần né tránh mọi cảm xúc khó khăn để tiến về phía trước.", "adversarial_tags": ["negation", "modality"]},
    {"sample_id": "extra-self-help-02", "domain": "SELF_HELP", "source": "If you cannot change the situation, change the next small step.", "preferred_output": "Nếu không thể thay đổi tình huống, hãy thay đổi bước nhỏ tiếp theo.", "adversarial_tags": ["condition", "modality"]},
    {"sample_id": "extra-self-help-03", "domain": "SELF_HELP", "source": "She told herself that resting was not a failure.", "preferred_output": "Cô tự nhủ rằng nghỉ ngơi không phải là thất bại.", "adversarial_tags": ["pronoun", "negation", "nominalization"]},
    {"sample_id": "extra-technical-01", "domain": "TECHNICAL", "source": "A transaction is considered idempotent when repeating it has the same effect.", "preferred_output": "Một giao dịch được xem là bất biến khi thực hiện lặp lại vẫn cho cùng một hiệu ứng.", "adversarial_tags": ["technical_definition", "passive"]},
    {"sample_id": "extra-technical-02", "domain": "TECHNICAL", "source": "The client must not retry a request after a non-retriable error.", "preferred_output": "Máy khách không được thử lại yêu cầu sau lỗi không thể thử lại.", "adversarial_tags": ["legal_obligation", "negation", "technical_definition"]},
    {"sample_id": "extra-technical-03", "domain": "TECHNICAL", "source": "When the queue is full, the worker falls back to a local buffer.", "preferred_output": "Khi hàng đợi đầy, worker chuyển sang bộ đệm cục bộ.", "adversarial_tags": ["condition", "phrasal_verb"]},
    {"sample_id": "extra-academic-01", "domain": "ACADEMIC", "source": "The association may reflect selection bias rather than a causal effect.", "preferred_output": "Mối liên hệ này có thể phản ánh sai lệch chọn mẫu thay vì hiệu ứng nhân quả.", "adversarial_tags": ["academic_hedging", "causality"]},
    {"sample_id": "extra-academic-02", "domain": "ACADEMIC", "source": "The hypothesis was not supported by the data collected in 2024.", "preferred_output": "Giả thuyết không được dữ liệu thu thập năm 2024 ủng hộ.", "adversarial_tags": ["passive", "negation", "date"]},
    {"sample_id": "extra-academic-03", "domain": "ACADEMIC", "source": "Although the effect is statistically significant, its practical importance remains uncertain.", "preferred_output": "Mặc dù hiệu ứng có ý nghĩa thống kê, tầm quan trọng thực tiễn của nó vẫn chưa chắc chắn.", "adversarial_tags": ["academic_hedging", "discourse_connector", "pronoun"]},
    {"sample_id": "extra-legal-01", "domain": "LEGAL", "source": "The provider shall not use the data for any purpose other than the services.", "preferred_output": "Nhà cung cấp không được sử dụng dữ liệu cho bất kỳ mục đích nào ngoài việc cung cấp dịch vụ.", "adversarial_tags": ["legal_obligation", "negation"]},
    {"sample_id": "extra-legal-02", "domain": "LEGAL", "source": "The notice is valid only if it is received before 5:00 p.m. on 1 October 2026.", "preferred_output": "Thông báo chỉ hợp lệ nếu được nhận trước 17 giờ ngày 1 tháng 10 năm 2026.", "adversarial_tags": ["legal_obligation", "condition", "date"]},
    {"sample_id": "extra-legal-03", "domain": "LEGAL", "source": "Nothing in this clause shall be construed as a waiver of either party's rights.", "preferred_output": "Không nội dung nào trong điều khoản này được hiểu là sự từ bỏ quyền của một trong hai bên.", "adversarial_tags": ["legal_obligation", "negation", "passive"]},
    {"sample_id": "extra-literature-01", "domain": "LITERATURE", "source": "The old house remembered every footstep, or so she believed.", "preferred_output": "Ngôi nhà cũ dường như nhớ từng bước chân, ít nhất là cô tin như vậy.", "adversarial_tags": ["idiom", "academic_hedging", "pronoun"]},
    {"sample_id": "extra-literature-02", "domain": "LITERATURE", "source": "He held his breath until the last light went out.", "preferred_output": "Anh nín thở cho đến khi vệt sáng cuối cùng tắt hẳn.", "adversarial_tags": ["collocation", "causality"]},
    {"sample_id": "extra-literature-03", "domain": "LITERATURE", "source": "The road split in two, and neither path promised a return.\nNeither sign pointed home.", "preferred_output": "Con đường tách làm đôi, mà chẳng lối nào hứa hẹn ngày trở lại.\nKhông tấm biển nào chỉ đường về nhà.", "adversarial_tags": ["idiom", "negation", "sentence_merge", "multi_paragraph"]},
]


def _infer_tags(source: str) -> List[str]:
    lowered = (source or "").casefold()
    tags = set()
    if re.search(r"\b(?:not|never|no|neither|nothing|cannot|can't)\b", lowered):
        tags.add("negation")
    if "not" in lowered and re.search(r"\b(?:no|never|without|not)\b.*\bnot\b", lowered):
        tags.add("double_negation")
    if re.search(r"\b(?:may|might|must|should|could|shall|would)\b", lowered):
        tags.add("modality")
    if re.search(r"\b(?:if|unless|provided|although|whether)\b", lowered):
        tags.add("condition")
    if re.search(r"\b(?:because|therefore|thus|resulted|so that|rather than)\b", lowered):
        tags.add("causality")
    if re.search(r"\$|%|\b\d+(?:\.\d+)?\b", source or ""):
        tags.add("percent" if "%" in (source or "") else "currency" if "$" in (source or "") else "date")
    if re.search(r"\b(?:which|who|she|he|they|it|her|him|their)\b", lowered):
        tags.add("pronoun")
    if re.search(r"\b(?:is|are|was|were|been|be)\s+\w+(?:ed|en)\b", lowered):
        tags.add("passive")
    if re.search(r"\b(?:tion|ment|ity|ance|ence)\b", lowered):
        tags.add("nominalization")
    if len((source or "").split()) >= 45:
        tags.add("long_paragraph")
    return sorted(tags & ADVERSARIAL_TAGS)


def _normalize_row(row: Dict[str, Any], source_name: str, index: int) -> Dict[str, Any]:
    source = str(row.get("source", "") or "")
    domain = str(row.get("domain", "GENERAL") or "GENERAL").upper()
    preferred = (
        row.get("preferred_output")
        or row.get("natural_candidate")
        or row.get("repair_candidate")
        or row.get("candidate")
        or ""
    )
    result = {
        "sample_id": str(row.get("sample_id") or row.get("id") or f"{source_name}-{index:04d}"),
        "domain": domain if domain in DOMAINS else "GENERAL",
        "source": source,
        "node_type": row.get("node_type", "paragraph"),
        "preferred_output": str(preferred or ""),
        "adversarial_tags": sorted(set(row.get("adversarial_tags") or []) | set(_infer_tags(source))),
        "evaluation_status": str(row.get("evaluation_status", "PENDING_HUMAN")),
    }
    for field_name in SCORE_FIELDS:
        value = row.get(field_name)
        result[field_name] = value if isinstance(value, (int, float)) and 1 <= value <= 5 else None
    result["source_fixture"] = source_name
    result["glossary"] = row.get("glossary", {})
    return result


def load_official_dataset(base_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Đọc 196 fixture cũ và bổ sung 24 mẫu adversarial, tổng cộng 220 mẫu."""
    root = base_dir or Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "translation_eval"
    paths = (
        root / "phase2_eval.jsonl",
        root / "vietnamese_naturalness_eval.jsonl",
        root / "phase3_semantic_eval.jsonl",
    )
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                normalized = _normalize_row(json.loads(line), path.stem, index)
                normalized["sample_id"] = f"{path.stem}:{normalized['sample_id']}"
                rows.append(normalized)
    rows.extend(_normalize_row(row, "p1_p2_adversarial", index) for index, row in enumerate(_EXTRA_SAMPLES, start=1))
    return rows


def validate_official_dataset(rows: Sequence[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    if len(rows) < 200:
        errors.append(f"dataset chỉ có {len(rows)} mẫu, cần tối thiểu 200")
    if {str(row.get("domain", "")).upper() for row in rows} != set(DOMAINS):
        errors.append("dataset phải phủ đủ tám domain")
    for index, row in enumerate(rows):
        for key in ("sample_id", "domain", "source", "preferred_output", *SCORE_FIELDS):
            if key not in row:
                errors.append(f"row {index} thiếu field {key}")
        for tag in row.get("adversarial_tags", []):
            if tag not in ADVERSARIAL_TAGS:
                errors.append(f"row {index} có tag không hợp lệ: {tag}")
    return errors
