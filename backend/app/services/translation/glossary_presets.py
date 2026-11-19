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
        # Dịch thử Pro Git: model giữ đúng các từ này như lập trình viên Việt vẫn viết, nhưng gate coi là chữ sót nên
        # 10/12 đoạn phải dịch lại rồi chờ duyệt. Đưa vào preset để model và gate cùng một quy ước.
        ("branch", "nhánh", ("branch",), "nhánh của git"),
        ("merge", "merge", ("hợp nhất",), "thao tác của git"),
        ("rebase", "rebase", (), "thao tác của git"),
        ("push", "push", ("đẩy",), "đẩy commit lên máy chủ"),
        ("pull", "pull", ("kéo về",), "kéo thay đổi về"),
        ("fetch", "fetch", ("tải về",), "thao tác của git"),
        ("stash", "stash", (), "cất tạm thay đổi trong git"),
        ("patch", "bản vá", ("patch",), "tệp chứa thay đổi mã nguồn"),
        ("diff", "diff", ("khác biệt",), "kết quả so sánh thay đổi"),
        ("server", "máy chủ", ("server",), ""),
        ("script", "script", ("kịch bản",), "tệp lệnh"),
        ("output", "đầu ra", ("output", "kết quả"), "kết quả lệnh in ra"),
        ("working directory", "thư mục làm việc", (), ""),
        # Dịch thử README thư viện Python: model giữ "framework" như lập trình viên Việt vẫn viết, hoặc dịch "khung làm việc".
        ("framework", "framework", ("khung",), "bộ khung phần mềm (web framework), không dịch là 'khung làm việc'"),
        ("binding", "binding", ("lớp liên kết",), "lớp bọc để gọi thư viện viết bằng ngôn ngữ khác (Python binding)"),
        # Dịch thử README thư viện Python: các từ này được giữ nguyên như lập trình viên Việt vẫn viết mà gate coi là chữ sót.
        ("container", "container", ("vùng chứa",), "môi trường chạy cô lập (Docker container)"),
        ("header", "header", ("phần đầu",), "phần đầu của yêu cầu/phản hồi HTTP hoặc của tệp"),
        ("cookie", "cookie", (), "dữ liệu trình duyệt lưu cho trang web"),
        ("proxy", "proxy", ("máy chủ trung gian",), ""),
        ("fork", "fork", ("bản tách",), "bản sao tách riêng của một dự án mã nguồn"),
        ("clone", "clone", ("sao chép",), "tạo bản sao kho mã git"),
        ("staging area", "vùng staging", ("staging area",), ""),
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
        # Dịch thử Darwin và William James: model chọn sai các thuật ngữ chuẩn này ("embryo" thành "thai nhi",
        # "larval and pupal stages" thành "giai đoạn trứng và bướm", "determinists" thành "những người duy vật").
        ("natural selection", "chọn lọc tự nhiên", (), ""),
        ("embryo", "phôi", (), "giai đoạn đầu của sinh vật, không phải 'thai nhi'"),
        ("larva", "ấu trùng", (), ""),
        ("larvae", "ấu trùng", (), ""),
        ("larval", "ấu trùng", (), ""),
        ("pupal", "nhộng", (), ""),
        ("crustacean", "giáp xác", ("động vật giáp xác",), ""),
        ("naturalist", "nhà tự nhiên học", (), "không phải 'nhà phân loại'"),
        ("rationalism", "chủ nghĩa duy lý", (), ""),
        ("empiricism", "chủ nghĩa kinh nghiệm", (), ""),
        ("determinism", "thuyết tất định", (), ""),
        ("determinist", "người theo thuyết tất định", (), "không phải 'người duy vật'"),
        ("idealism", "chủ nghĩa duy tâm", (), ""),
        ("pragmatism", "chủ nghĩa thực dụng", (), ""),
        ("pragmatist", "nhà thực dụng", ("người theo chủ nghĩa thực dụng",), ""),
    ],
}
# Thuật ngữ kinh tế học cổ điển hay gặp trong sách tài chính, kinh tế (dịch thử Adam Smith).
PRESETS["FINANCE"] += [
    ("annuity", "niên kim", ("khoản trợ cấp hằng năm",), ""),
    ("bank notes", "giấy bạc ngân hàng", ("tiền giấy ngân hàng",), "không phải 'tiền mặt'"),
    ("exchequer", "kho bạc", ("Bộ Tài chính",), ""),
    ("shilling", "shilling", ("silinh",), "đơn vị tiền Anh"),
]


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
