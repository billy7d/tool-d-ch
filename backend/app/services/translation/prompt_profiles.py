from dataclasses import dataclass
from typing import Dict, List


STYLE_PACK_VERSION = "style-packs-v1"


@dataclass(frozen=True)
class StylePack:
    domain: str
    register: str
    instructions: str
    forbidden: str


STYLE_PACKS: Dict[str, StylePack] = {
    "GENERAL": StylePack("GENERAL", "neutral, accessible", "Write clear, natural and accessible Vietnamese. Prefer common Vietnamese syntax and concise verbs.", "Avoid needless academic or Sino-Vietnamese wording and literal English syntax."),
    "BUSINESS": StylePack("BUSINESS", "professional", "Use concise professional Vietnamese and established business terminology while keeping the prose readable.", "Do not turn ordinary prose into stiff corporate jargon."),
    "FINANCE": StylePack("FINANCE", "precise", "Prioritize standard financial terminology and preserve every number, ratio, metric, unit and degree of certainty.", "Do not naturalize wording in a way that changes financial meaning."),
    "SELF_HELP": StylePack("SELF_HELP", "accessible, encouraging", "Use friendly, direct and motivating Vietnamese with readable sentence length.", "Avoid research-like prose, heavy vocabulary and exaggerated promises."),
    "TECHNICAL": StylePack("TECHNICAL", "precise, practical", "Preserve technical accuracy and standard terminology. Keep APIs, commands, identifiers, class and function names unchanged.", "Do not translate code or invent technical explanations."),
    "ACADEMIC": StylePack("ACADEMIC", "formal, scholarly", "Use formal, precise, neutral Vietnamese appropriate for scholarly publication.", "Do not make the sentence harder than the source or add claims."),
    "LEGAL": StylePack("LEGAL", "formal, conservative", "Preserve scope, modality, conditions, obligations, prohibitions and defined terms exactly. Restructure conservatively.", "Do not soften or strengthen legal force."),
    "LITERATURE": StylePack("LITERATURE", "voice-driven", "Preserve narrative voice, rhythm, dialogue, characterization and emotional nuance in idiomatic Vietnamese.", "Do not embellish facts or rewrite the plot."),
}


FEW_SHOTS: Dict[str, List[dict]] = {
    "GENERAL": [{"node_type": "paragraph", "source": "It is important to note that the plan was changed in order to reduce delays.", "target": "Cần lưu ý rằng kế hoạch đã được điều chỉnh để giảm chậm trễ."}],
    "BUSINESS": [{"node_type": "paragraph", "source": "The company has taken significant steps toward improving operational efficiency.", "target": "Công ty đã có nhiều thay đổi quan trọng để nâng cao hiệu quả hoạt động."}],
    "FINANCE": [{"node_type": "paragraph", "source": "Revenue rose 12%, while the debt-to-equity ratio remained at 0.8.", "target": "Doanh thu tăng 12%, trong khi hệ số nợ trên vốn chủ sở hữu vẫn ở mức 0,8."}],
    "SELF_HELP": [{"node_type": "paragraph", "source": "Taking action is often more useful than waiting for the perfect moment.", "target": "Hành động thường hữu ích hơn là chờ đến thời điểm hoàn hảo."}],
    "TECHNICAL": [{"node_type": "paragraph", "source": "Call the HTTP API with curl, then parse the JSON response in Python.", "target": "Gọi API HTTP bằng curl, sau đó phân tích phản hồi JSON trong Python."}],
    "ACADEMIC": [{"node_type": "paragraph", "source": "The optimization of the structure resulted in a measurable improvement.", "target": "Việc tối ưu cấu trúc mang lại mức cải thiện có thể đo lường được."}],
    "LEGAL": [{"node_type": "paragraph", "source": "The licensee may terminate this Agreement only if written notice is provided 30 days in advance.", "target": "Bên được cấp phép chỉ có thể chấm dứt Thỏa thuận này nếu gửi thông báo bằng văn bản trước 30 ngày."}],
    "LITERATURE": [{"node_type": "dialogue", "source": "\"You came back,\" she said, barely above a whisper.", "target": "“Anh đã về,” cô khẽ nói, giọng chỉ như thì thầm."}],
}


def get_style_pack(domain: str) -> StylePack:
    return STYLE_PACKS.get((domain or "GENERAL").upper(), STYLE_PACKS["GENERAL"])


def select_few_shots(domain: str, mode: str, node_type: str, limit: int = 1) -> List[dict]:
    candidates = FEW_SHOTS.get((domain or "GENERAL").upper(), FEW_SHOTS["GENERAL"])
    exact = [item for item in candidates if item["node_type"] == node_type]
    return (exact or candidates)[:max(0, limit)]
