"""Bộ thuật ngữ mẫu theo lĩnh vực, dùng như thuật ngữ MỀM (gợi ý trong prompt, không chặn bản dịch).

Chạy thật với qwen2.5:7b, các lỗi như "licensee" -> "Nhà sử dụng bản quyền" hay "build command" ->
"lệnh xây dựng" lặp lại ở mọi vòng. Thuật ngữ của project (Bước 4) luôn được ưu tiên hơn preset.
Bật/tắt qua style_guide["glossary_presets"]: danh sách lĩnh vực, mặc định là lĩnh vực của tài liệu.
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple

GLOSSARY_PRESETS_VERSION = "glossary-presets-v1"

# (nguồn, đích ưu tiên, biến thể chấp nhận, gợi ý nghĩa)
_Preset = Tuple[str, str, Tuple[str, ...], str]

PRESETS: Dict[str, List[_Preset]] = {
    "LEGAL": [
        ("licensee", "bên được cấp phép", ("bên nhận giấy phép",), "bên nhận quyền sử dụng theo giấy phép"),
        ("licensor", "bên cấp phép", (), ""),
        ("party", "bên", ("các bên",), "bên tham gia hợp đồng"),
        ("either party", "bất kỳ bên nào", ("một trong hai bên",), "không phải 'cả hai bên'"),
        ("waiver", "sự từ bỏ quyền", ("việc khước từ quyền",), ""),
        ("indemnify", "bồi hoàn", ("bồi thường",), ""),
        ("liability", "trách nhiệm pháp lý", (), ""),
        ("breach", "vi phạm", (), ""),
        ("termination", "việc chấm dứt", ("chấm dứt",), ""),
        ("governing law", "luật điều chỉnh", (), ""),
        ("confidential information", "thông tin mật", ("thông tin bảo mật",), ""),
        ("warranty", "bảo đảm", ("bảo hành",), "cam kết trong hợp đồng"),
        ("written notice", "thông báo bằng văn bản", (), ""),
    ],
    "TECHNICAL": [
        ("build command", "lệnh build", (), "giữ 'build' như lập trình viên Việt quen dùng"),
        ("build", "build", ("bản build",), "biên dịch/đóng gói phần mềm, không dịch là 'xây dựng'"),
        ("deploy", "triển khai", ("deploy",), ""),
        ("repository", "repository", ("kho mã",), ""),
        ("commit", "commit", (), "thao tác của git"),
        ("pull request", "pull request", (), ""),
        ("endpoint", "endpoint", (), ""),
        ("cache", "bộ nhớ đệm", ("cache",), ""),
        ("client", "client", ("máy khách",), "phần mềm/dịch vụ gọi API, không phải 'khách hàng'"),
        ("dependency", "dependency", ("phụ thuộc",), "gói thư viện"),
        ("backward compatible", "tương thích ngược", (), ""),
    ],
    "FINANCE": [
        ("gross profit", "lợi nhuận gộp", (), ""),
        ("net cash flow", "dòng tiền thuần", ("dòng tiền ròng",), ""),
        ("operating costs", "chi phí hoạt động", ("chi phí vận hành",), ""),
        ("revenue", "doanh thu", (), ""),
        ("equity", "vốn chủ sở hữu", (), ""),
        ("debt-to-equity ratio", "hệ số nợ trên vốn chủ sở hữu", (), ""),
        ("basis points", "điểm cơ bản", (), ""),
        ("dividend", "cổ tức", (), ""),
        ("percentage points", "điểm phần trăm", (), ""),
        ("target rate", "lãi suất mục tiêu", (), ""),
    ],
    "BUSINESS": [
        ("stakeholders", "các bên liên quan", (), ""),
        ("operational efficiency", "hiệu quả hoạt động", (), ""),
        ("product launch", "ra mắt sản phẩm", (), ""),
        ("management", "ban lãnh đạo", ("ban quản lý",), "khi chỉ nhóm điều hành công ty"),
        ("market share", "thị phần", (), ""),
    ],
    "ACADEMIC": [
        ("et al.", "và cộng sự", (), ""),
        ("findings", "các phát hiện", ("kết quả nghiên cứu",), ""),
        ("measurable", "đo lường được", ("có thể đo lường",), "không phải 'đáng kể'"),
        ("significant", "có ý nghĩa thống kê", ("đáng kể",), "trong ngữ cảnh thống kê"),
        ("causality", "quan hệ nhân quả", (), ""),
    ],
}


def presets_for(style_guide: Optional[Dict[str, Any]], document_type: str) -> List[str]:
    configured = (style_guide or {}).get("glossary_presets")
    if configured is None:
        return [str(document_type or "GENERAL").upper()] if str(document_type or "").upper() in PRESETS else []
    return [str(domain).upper() for domain in configured if str(domain).upper() in PRESETS]


def preset_entries(domains: Iterable[str], existing_sources: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Entry cùng shape với GlossaryService.to_context_entry, lock SOFT, bỏ term project đã có."""
    taken = {str(source).casefold() for source in existing_sources}
    entries: List[Dict[str, Any]] = []
    for domain in domains:
        for source, target, variants, sense in PRESETS.get(domain, []):
            if source.casefold() in taken:
                continue
            taken.add(source.casefold())
            entries.append({
                "id": f"preset:{domain}:{source}",
                "source_term": source,
                "target_term": target,
                "preferred_target": target,
                "allowed_variants": list(variants),
                "sense_hint": sense,
                "domain": domain,
                "part_of_speech": "",
                "preserve_original": False,
                "lock_level": "SOFT",
                "locked": False,
                "notes": GLOSSARY_PRESETS_VERSION,
            })
    return entries
