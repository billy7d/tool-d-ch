"""Học quy tắc văn phong cấp project từ các chỉnh sửa của người biên tập.

Khi người dùng duyệt một node mà bản cuối khác bản AI, các cụm bị thay thế ngắn
(tối đa vài từ) được ghi lại thành cặp "tránh -> dùng". Một cặp chỉ có hiệu lực khi
lặp lại ở đủ số node khác nhau, để một lần sửa riêng lẻ không thành luật chung.
"""

import difflib
import re
import unicodedata
import uuid
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import StyleRuleModel, TranslationModel


STYLE_RULES_VERSION = "style-rules-v1"
ACTIVATION_THRESHOLD = 2
MAX_RULE_WORDS = 4
MAX_PROMPT_RULES = 12

_WORD = re.compile(r"[^\s]+")
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
_HAS_DIGIT = re.compile(r"\d")


def _words(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFC", text or "")
    words = []
    for token in _WORD.findall(normalized):
        cleaned = _EDGE_PUNCT.sub("", token)
        if cleaned:
            words.append(cleaned.casefold())
    return words


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", unicodedata.normalize("NFC", text or ""), re.IGNORECASE))


class StyleRuleService:
    @staticmethod
    def _edit_hunks(opcodes) -> List[Tuple[int, int, int, int]]:
        """Gộp các thay đổi chỉ cách nhau một từ giữ nguyên.

        Diff tối thiểu của "một cách rõ ràng" -> "rõ" là hai lần xóa quanh chữ "rõ";
        gộp lại mới ra đúng cụm người biên tập đã thay.
        """
        hunks: List[List[int]] = []
        pending_equal = None
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                pending_equal = (i1, i2, j1, j2)
                continue
            if hunks and pending_equal and pending_equal[1] - pending_equal[0] <= 1 and hunks[-1][1] == pending_equal[0]:
                hunks[-1][1], hunks[-1][3] = i2, j2
            else:
                hunks.append([i1, i2, j1, j2])
            pending_equal = None
        return [tuple(item) for item in hunks]

    @staticmethod
    def extract_edit_rules(ai_text: str, edited_text: str) -> List[Tuple[str, str]]:
        """Trả về các cặp (cụm AI, cụm người sửa) ngắn, bỏ qua thay đổi số liệu và nội dung dài."""
        before, after = _words(ai_text), _words(edited_text)
        if not before or not after or before == after:
            return []
        rules: List[Tuple[str, str]] = []
        matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
        # Bản sửa khác quá nhiều là viết lại cả đoạn, không phải chỉnh văn phong cục bộ.
        if matcher.ratio() < 0.5:
            return []
        for i1, i2, j1, j2 in StyleRuleService._edit_hunks(matcher.get_opcodes()):
            removed, inserted = before[i1:i2], after[j1:j2]
            if not removed or len(removed) > MAX_RULE_WORDS or len(inserted) > MAX_RULE_WORDS:
                continue
            avoid, prefer = " ".join(removed), " ".join(inserted)
            if _HAS_DIGIT.search(avoid) or _HAS_DIGIT.search(prefer) or avoid == prefer:
                continue
            # Xóa một từ đơn lẻ (vd "các") quá chung chung để thành luật.
            if len(removed) == 1 and not inserted:
                continue
            rules.append((avoid, prefer))
        return rules

    @staticmethod
    def latest_ai_translation(db: Session, node_id: str) -> str:
        row = db.query(TranslationModel).filter(
            TranslationModel.node_id == node_id,
        ).order_by(TranslationModel.created_at.desc()).first()
        return str(row.translated_text or "") if row else ""

    @classmethod
    def observe_edit(cls, db: Session, project_id: str, node_id: str, ai_text: str, edited_text: str, commit: bool = True) -> List[StyleRuleModel]:
        touched = []
        for avoid, prefer in cls.extract_edit_rules(ai_text, edited_text):
            rule = db.query(StyleRuleModel).filter(
                StyleRuleModel.project_id == project_id,
                StyleRuleModel.avoid == avoid,
                StyleRuleModel.prefer == prefer,
            ).first()
            if not rule:
                rule = StyleRuleModel(
                    id=str(uuid.uuid4()), project_id=project_id, avoid=avoid, prefer=prefer,
                    node_ids=[], occurrences=0, status="CANDIDATE",
                )
                db.add(rule)
            node_ids = list(rule.node_ids or [])
            if node_id in node_ids:
                continue
            node_ids.append(node_id)
            rule.node_ids = node_ids
            rule.occurrences = len(node_ids)
            if rule.status == "CANDIDATE" and rule.occurrences >= ACTIVATION_THRESHOLD:
                rule.status = "ACTIVE"
            touched.append(rule)
        if commit:
            db.commit()
        else:
            db.flush()
        return touched

    @classmethod
    def observe_approved_node(cls, db: Session, project_id: str, node) -> List[StyleRuleModel]:
        edited = str(getattr(node, "translated_content", "") or "")
        ai_text = cls.latest_ai_translation(db, node.id)
        if not ai_text or not edited or ai_text.strip() == edited.strip():
            return []
        return cls.observe_edit(db, project_id, node.id, ai_text, edited)

    @staticmethod
    def active_rules(db: Session, project_id: str, limit: int = MAX_PROMPT_RULES) -> List[StyleRuleModel]:
        return db.query(StyleRuleModel).filter(
            StyleRuleModel.project_id == project_id,
            StyleRuleModel.status == "ACTIVE",
        ).order_by(StyleRuleModel.occurrences.desc(), StyleRuleModel.avoid).limit(limit).all()

    @staticmethod
    def list_rules(db: Session, project_id: str) -> List[StyleRuleModel]:
        return db.query(StyleRuleModel).filter(
            StyleRuleModel.project_id == project_id,
        ).order_by(StyleRuleModel.occurrences.desc(), StyleRuleModel.avoid).all()

    @staticmethod
    def set_status(db: Session, project_id: str, rule_id: str, status: str) -> Optional[StyleRuleModel]:
        normalized = str(status or "").upper()
        if normalized not in {"ACTIVE", "DISABLED", "CANDIDATE"}:
            raise ValueError("Trạng thái quy tắc chỉ nhận ACTIVE, DISABLED hoặc CANDIDATE.")
        rule = db.query(StyleRuleModel).filter(
            StyleRuleModel.project_id == project_id, StyleRuleModel.id == rule_id,
        ).first()
        if rule:
            rule.status = normalized
            db.commit()
        return rule

    @staticmethod
    def prompt_lines(rules: List[StyleRuleModel]) -> List[str]:
        lines = []
        for rule in rules:
            if rule.prefer:
                lines.append(f'- Tránh "{rule.avoid}", dùng "{rule.prefer}" ({rule.occurrences} lần sửa).')
            else:
                lines.append(f'- Tránh "{rule.avoid}"; thường lược bỏ được ({rule.occurrences} lần sửa).')
        return lines

    @staticmethod
    def lint(rules: List[StyleRuleModel], translated_text: str) -> List[Dict[str, str]]:
        """Hint cùng shape với VietnameseStyleLint để critic/rewrite dùng chung."""
        hints = []
        for rule in rules:
            if _contains_phrase(translated_text, rule.avoid):
                hints.append({
                    "rule": "PROJECT_STYLE_RULE",
                    "type": "COLLOCATION",
                    "severity": "WARNING",
                    "target_span": rule.avoid,
                    "message": f'Người biên tập đã nhiều lần sửa "{rule.avoid}" trong project này.',
                    "suggestion": f'Dùng "{rule.prefer}".' if rule.prefer else "Lược bỏ cụm này nếu nghĩa không đổi.",
                })
        return hints

    @staticmethod
    def to_dict(rule: StyleRuleModel) -> Dict[str, object]:
        return {
            "id": rule.id,
            "avoid": rule.avoid,
            "prefer": rule.prefer,
            "occurrences": rule.occurrences,
            "status": rule.status,
        }
