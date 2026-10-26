from dataclasses import dataclass
from typing import Dict, List

from app.services.translation.few_shot_library import (
    FEW_SHOT_LIBRARY_VERSION,
    SUPPORTED_DOMAINS,
    curated_examples_as_dict,
)
from app.services.translation.few_shot_selector import select_few_shots


STYLE_PACK_VERSION = "style-packs-v2"


@dataclass(frozen=True)
class StylePack:
    domain: str
    register: str
    instructions: str
    forbidden: str
    # Quy ước trình bày tiếng Việt của miền (dấu câu, số, ngày, lời thoại).
    conventions: str = ""


STYLE_PACKS: Dict[str, StylePack] = {
    "GENERAL": StylePack("GENERAL", "neutral, accessible", "Write clear, natural and accessible Vietnamese. Prefer common Vietnamese syntax and concise verbs.", "Avoid needless academic or Sino-Vietnamese wording and literal English syntax.", "Dùng ngoặc kép “…” cho trích dẫn. Số thập phân dùng dấu phẩy (0,8), phân cách hàng nghìn dùng dấu chấm (1.000). Ngày viết 'ngày 12 tháng 3 năm 2026'."),
    "BUSINESS": StylePack("BUSINESS", "professional", "Use concise professional Vietnamese and established business terminology while keeping the prose readable.", "Do not turn ordinary prose into stiff corporate jargon.", "Dùng ngoặc kép “…” cho trích dẫn. Số thập phân dùng dấu phẩy (0,8), phân cách hàng nghìn dùng dấu chấm (1.000). Ngày viết 'ngày 12 tháng 3 năm 2026'. Giữ nguyên tên công ty, sản phẩm và chức danh tiếng Anh đã thành thông dụng (CEO, KPI)."),
    "FINANCE": StylePack("FINANCE", "precise", "Prioritize standard financial terminology and preserve every number, ratio, metric, unit and degree of certainty.", "Do not naturalize wording in a way that changes financial meaning.", "Dùng ngoặc kép “…” cho trích dẫn. Số thập phân dùng dấu phẩy (0,8), phân cách hàng nghìn dùng dấu chấm (1.000). Ngày viết 'ngày 12 tháng 3 năm 2026'. Phần trăm viết liền '12%'. Quý viết 'quý II'. Đơn vị tiền tệ giữ ký hiệu gốc ($, €) đặt đúng vị trí của nguồn."),
    "SELF_HELP": StylePack("SELF_HELP", "accessible, encouraging", "Use friendly, direct and motivating Vietnamese with readable sentence length.", "Avoid research-like prose, heavy vocabulary and exaggerated promises.", "Dùng ngoặc kép “…” cho trích dẫn. Số thập phân dùng dấu phẩy (0,8), phân cách hàng nghìn dùng dấu chấm (1.000). Ngày viết 'ngày 12 tháng 3 năm 2026'. Gọi người đọc nhất quán theo sổ xưng hô; câu ngắn, giọng gần gũi."),
    "TECHNICAL": StylePack("TECHNICAL", "precise, practical", "Preserve technical accuracy and standard terminology. Keep APIs, commands, identifiers, class and function names unchanged.", "Do not translate code or invent technical explanations.", 'Giữ nguyên code, lệnh, đường dẫn, tên hàm và identifier trong `…`. Giữ thuật ngữ tiếng Anh mà lập trình viên Việt quen dùng (API, log, request, commit) khi không có glossary. Số thập phân dùng dấu phẩy trong văn xuôi, nhưng giữ nguyên trong code.'),
    "ACADEMIC": StylePack("ACADEMIC", "formal, scholarly", "Use formal, precise, neutral Vietnamese appropriate for scholarly publication.", "Do not make the sentence harder than the source or add claims.", "Dùng ngoặc kép “…” cho trích dẫn. Số thập phân dùng dấu phẩy (0,8), phân cách hàng nghìn dùng dấu chấm (1.000). Ngày viết 'ngày 12 tháng 3 năm 2026'. 'et al.' dịch là 'và cộng sự'. Giữ nguyên trích dẫn (Tác giả, năm) và số hiệu bảng, hình."),
    "LEGAL": StylePack("LEGAL", "formal, conservative", "Preserve scope, modality, conditions, obligations, prohibitions and defined terms exactly. Restructure conservatively.", "Do not soften or strengthen legal force.", "Viết hoa 'Điều', 'Khoản', 'Mục' khi đi kèm số hiệu. Giữ viết hoa thuật ngữ được định nghĩa (Thỏa thuận, Bên A). 'shall' dịch là 'phải', 'may' là 'có thể/được quyền' theo ngữ cảnh. Dùng ngoặc kép “…” cho trích dẫn. Số thập phân dùng dấu phẩy (0,8), phân cách hàng nghìn dùng dấu chấm (1.000). Ngày viết 'ngày 12 tháng 3 năm 2026'."),
    "LITERATURE": StylePack("LITERATURE", "voice-driven", "Preserve narrative voice, rhythm, dialogue, characterization and emotional nuance in idiomatic Vietnamese.", "Do not embellish facts or rewrite the plot.", "Lời thoại dùng ngoặc kép “…” hoặc gạch đầu dòng, giữ nhất quán với phần đã dịch. Dấu phẩy và lời dẫn đặt sau ngoặc: “Anh về rồi”, cô nói. Dùng dấu chấm lửng '…'. Xưng hô giữa nhân vật theo sổ xưng hô và giữ ổn định suốt truyện."),
}


# Giữ tên export cũ cho benchmark P0/P1, nhưng dữ liệu nay đến từ thư viện curated.
FEW_SHOTS: Dict[str, List[dict]] = {
    domain: curated_examples_as_dict(domain) for domain in SUPPORTED_DOMAINS
}


def get_style_pack(domain: str) -> StylePack:
    return STYLE_PACKS.get((domain or "GENERAL").upper(), STYLE_PACKS["GENERAL"])


# `select_few_shots` được import từ few_shot_selector để các caller cũ vẫn dùng
# đúng hàm nhưng đã có scoring theo source pattern và giới hạn tối đa bốn ví dụ.
