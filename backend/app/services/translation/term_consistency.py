"""Chọn một cách dịch cho các khái niệm chính của cuốn sách, một lần trước khi dịch.

Mỗi đoạn được dịch riêng nên cùng một khái niệm ra nhiều cách: trong một bản dịch thật, "grace" thành ơn, ân huệ,
ân sủng, ân điển; "psychotherapy" thành bốn cụm khác nhau; "ego boundaries" thành "ranh giới của tôi". Bước này
tìm các từ và cụm lặp lại nhiều, cho model chọn những cái là khái niệm của sách rồi chọn một cách dịch dựa trên
nghĩa từ điển và câu ví dụ lấy từ chính cuốn sách. Kết quả lưu thành thuật ngữ MỀM loại AUTO trong glossary: lượt
dịch chỉ được gợi ý, không bị ép, và người dùng sửa hoặc xoá được trong màn hình glossary.

Tốn vài lệnh gọi model cho cả cuốn; không thêm lệnh gọi nào cho từng đoạn.
"""
import hashlib
import json
import logging
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from app.services.translation.quality_gate import _VI_SYLLABLE

logger = logging.getLogger(__name__)
AUTO_TERMS_VERSION = "1"
AUTO_CATEGORY = "AUTO"
MIN_PARAGRAPHS = 8          # một từ phải có mặt ở ít nhất chừng này đoạn mới đáng thống nhất
MIN_PHRASE_PARAGRAPHS = 6
MAX_WORD_CANDIDATES = 90
MAX_PHRASE_CANDIDATES = 40
MAX_TERMS = 30
EXAMPLES_PER_TERM = 3
EXAMPLE_CHARS = 220

_WORD = re.compile(r"[A-Za-z]+(?:[’'-][A-Za-z]+)*")
_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
_VIETNAMESE_LETTER = re.compile(r"[à-ỹđ]", re.IGNORECASE)
_FOREIGN_SCRIPT = re.compile(r"[Ѐ-ԯᄀ-ᇿ⺀-鿿가-힯豈-﫿＀-￯]")
# Từ chức năng và từ thường ngày: cách dịch đổi theo câu, thống nhất máy móc chỉ làm văn cứng.
_COMMON = frozenset("""
a about above after again against all almost alone along already also although always am among an and another any
anyone anything are around as at away back be became because become becomes been before began begin behind being
below best better between both but by came can cannot come comes could day days did do does doing done down during
each either else end enough even ever every everyone everything example fact far feel feeling feels felt few find
first for found from further gave get gets getting give given gives go goes going gone good got great had hand has
have having he her here hers herself high him himself his how however if in indeed instead into is it its itself
just keep kind knew know known knows last later least left less let life like likely little long look looked looking
made make makes making man many may me mean means men might more most much must my myself need needs never new next
no nor not nothing now of off often old on once one only or order other others our ourselves out over own part
people perhaps person place point put quite rather really right said same saw say says second see seem seemed seems
seen several shall she should simply since so some someone something sometimes soon still such take taken takes
taking tell than that the their them themselves then there therefore these they thing things think thinking this
those though thought three through thus time times to today together told too took toward towards two under until
up upon us use used using very want wanted wants was way ways we well went were what whatever when where whether
which while who whole whom whose why will with within without woman women word words work would year years yet you
your yours yourself able ask asked call called case course different does early fact felt general hard help
important least less matter moment nature number often possible present problem question reason result sense set
side small sort state true turn usually various whole world
""".split())


def _tokens(text: str) -> List[str]:
    return _WORD.findall(text or "")


def mine_candidates(paragraphs: Sequence[str], headings: Sequence[str] = ()) -> List[Tuple[str, int]]:
    """Từ và cụm hai từ xuất hiện ở nhiều đoạn, kèm số đoạn. Tên riêng (viết hoa giữa câu) không tính."""
    words: Counter = Counter()
    phrases: Counter = Counter()
    for paragraph in paragraphs:
        tokens = _tokens(paragraph)
        seen_words, seen_phrases = set(), set()
        for index, token in enumerate(tokens):
            if not token.isalpha() or not token.islower() or len(token) < 3 or token in _COMMON:
                continue
            seen_words.add(token)
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            if following.isalpha() and following.islower() and len(following) >= 3 and following not in _COMMON:
                seen_phrases.add(f"{token} {following}")
        words.update(word for word in seen_words if len(word) >= 4)
        phrases.update(seen_phrases)
    # Số nhiều gộp vào số ít khi cả hai cùng có: "boundary"/"boundaries" được tính riêng vì TermMatcher không nối chúng.
    for word in list(words):
        if word.endswith("s") and not word.endswith("ss") and word[:-1] in words:
            words[word[:-1]] += words.pop(word)
    heading_words = {token.lower() for heading in headings for token in _tokens(heading)}

    def score(item: Tuple[str, int]) -> float:
        term, count = item
        return count * (1.5 if any(part in heading_words for part in term.split()) else 1.0)

    top_words = sorted(((w, c) for w, c in words.items() if c >= MIN_PARAGRAPHS), key=score, reverse=True)[:MAX_WORD_CANDIDATES]
    top_phrases = sorted(((p, c) for p, c in phrases.items() if c >= MIN_PHRASE_PARAGRAPHS), key=score, reverse=True)[:MAX_PHRASE_CANDIDATES]
    return top_phrases + top_words


def examples_for(term: str, paragraphs: Sequence[str]) -> List[str]:
    """Vài câu chứa thuật ngữ, lấy rải từ đầu, giữa và cuối sách."""
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    sentences = [
        sentence.strip() for paragraph in paragraphs if pattern.search(paragraph)
        for sentence in _SENTENCE.findall(paragraph) if pattern.search(sentence)
    ]
    if len(sentences) > EXAMPLES_PER_TERM:
        step = (len(sentences) - 1) / (EXAMPLES_PER_TERM - 1)
        sentences = [sentences[round(index * step)] for index in range(EXAMPLES_PER_TERM)]
    return [sentence[:EXAMPLE_CHARS] for sentence in sentences]


def valid_rendering(term: str, rendering: str) -> bool:
    """Chỉ nhận cụm tiếng Việt ngắn; model trả lại chính từ tiếng Anh, câu giải thích hay chữ lạ thì bỏ."""
    rendering = (rendering or "").strip()
    if not rendering or len(rendering) > 40 or len(rendering.split()) > 5:
        return False
    if _FOREIGN_SCRIPT.search(rendering) or re.search(r"[.;:!?()/\n]", rendering):
        return False
    if rendering.casefold() == term.casefold():
        return False
    # Từ không dấu phải ghép được thành âm tiết tiếng Việt; "therapia tâm lý" là từ model tự chế.
    if any(word.isascii() and not _VI_SYLLABLE.match(word.lower()) for word in rendering.split()):
        return False
    return bool(_VIETNAMESE_LETTER.search(rendering))


_SELECT_SYSTEM = (
    "You are preparing the terminology sheet for an English-to-Vietnamese book translation. From the candidate list, "
    "select the terms that are KEY CONCEPTS or domain terminology of this book: abstract concepts, technical terms and "
    "recurring set phrases that must get ONE consistent Vietnamese rendering across the whole book. Do NOT select "
    "everyday words whose natural translation changes with the sentence. Copy the selected terms exactly as written. "
    f"Return JSON {{\"terms\": [\"...\"]}} with at most {MAX_TERMS} items, most important first."
)
_RENDER_SYSTEM = (
    "You are a senior English-to-Vietnamese book translator. You get one English term and example sentences from the "
    "book that use it. First translate every example sentence into natural Vietnamese, rendering the term the SAME way "
    "in each one, with the standard Vietnamese term of the field. Then give that rendering alone: a short Vietnamese "
    "phrase of 1-5 words exactly as it appears in your translations, no English. "
    "Return JSON {\"translations\": [\"...\"], \"rendering\": \"...\"}."
)
_SELECT_SCHEMA = {
    "type": "object", "required": ["terms"],
    "properties": {"terms": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_TERMS}},
}
_RENDER_SCHEMA = {
    "type": "object", "required": ["translations", "rendering"],
    "properties": {"translations": {"type": "array", "items": {"type": "string"}}, "rendering": {"type": "string"}},
}


def attested(rendering: str, translations: Sequence[str]) -> bool:
    """Cách dịch phải có thật trong phần lớn các câu model vừa dịch; không thì đó là câu trả lời đoán ngoài ngữ cảnh."""
    needle = rendering.casefold()
    hits = sum(needle in str(translation).casefold() for translation in translations)
    return len(translations) >= 2 and hits * 3 >= len(translations) * 2


def choose_terms(
    provider: Any, model_name: str, paragraphs: Sequence[str], headings: Sequence[str] = (),
) -> List[Dict[str, str]]:
    """Trả về [{"source_term", "target_term"}]; rỗng khi sách quá ngắn hoặc provider không hỗ trợ."""
    json_task = getattr(provider, "json_task", None)
    candidates = mine_candidates(paragraphs, headings)
    if json_task is None or len(candidates) < 5:
        return []
    listing = "\n".join(f"- {term} ({count})" for term, count in candidates)
    selected = json_task(
        _SELECT_SYSTEM,
        "CHAPTER HEADINGS:\n" + "\n".join(list(headings)[:40])
        + "\n\nCANDIDATES (term, number of paragraphs it appears in):\n" + listing,
        _SELECT_SCHEMA, model=model_name, num_predict=400,
    )
    known = {term for term, _ in candidates}
    terms = list(dict.fromkeys(
        str(term).strip().lower() for term in (selected or {}).get("terms") or [] if str(term).strip().lower() in known
    ))[:MAX_TERMS]
    chosen: List[Dict[str, str]] = []
    for term in terms:
        examples = examples_for(term, paragraphs)
        if len(examples) < 2:
            continue
        numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(examples, 1))
        answer = json_task(_RENDER_SYSTEM, f"TERM: {term}\n{numbered}", _RENDER_SCHEMA, model=model_name, num_predict=500) or {}
        rendering = str(answer.get("rendering", "")).strip()
        if valid_rendering(term, rendering) and attested(rendering, answer.get("translations") or []):
            chosen.append({"source_term": term, "target_term": rendering})
    return chosen


def ensure_auto_terms(db: Any, project_id: str, provider: Any, model_name: str, cache_dir: Path) -> int:
    """Chạy một lần cho mỗi nội dung sách. Thuật ngữ người dùng đã có thì không đụng tới. Trả về số thuật ngữ thêm mới."""
    from app.db.models import GlossaryModel, NodeModel

    # Mặc định tắt: chạy thử trên sách thật, qwen2.5:7b chọn sai khoảng một phần ba ("unconscious" thành "tâm thức"),
    # mà một cách dịch sai áp cho cả cuốn thì tệ hơn là lệch nhau. Bật bằng TOOL_DICH_AUTO_TERMS=1 khi dùng model tốt hơn.
    if os.environ.get("TOOL_DICH_AUTO_TERMS", "0") != "1" or getattr(provider, "json_task", None) is None:
        return 0
    rows = db.query(NodeModel.node_type, NodeModel.content).filter(
        NodeModel.project_id == project_id, NodeModel.node_type.in_(["paragraph", "heading"]),
    ).order_by(NodeModel.order_index).all()
    paragraphs = [content for node_type, content in rows if node_type == "paragraph" and content]
    headings = [content for node_type, content in rows if node_type == "heading" and content]
    digest = hashlib.sha256(("\n".join(paragraphs) + AUTO_TERMS_VERSION).encode("utf-8")).hexdigest()
    marker = Path(cache_dir) / "auto_terms.json"
    try:
        if json.loads(marker.read_text(encoding="utf-8")).get("source_hash") == digest:
            return 0  # đã chạy cho nội dung này; thuật ngữ người dùng xoá đi không được tự thêm lại
    except (OSError, ValueError):
        pass
    try:
        chosen = choose_terms(provider, model_name, paragraphs, headings)
    except Exception as exc:
        logger.warning("Không chọn được thuật ngữ tự động: %s", exc)
        return 0
    existing = {
        str(source).casefold()
        for (source,) in db.query(GlossaryModel.source_term).filter(GlossaryModel.project_id == project_id).all()
    }
    added = 0
    for item in chosen:
        if item["source_term"].casefold() in existing:
            continue
        db.add(GlossaryModel(
            id=f"auto_{uuid.uuid4().hex[:12]}", project_id=project_id, source_term=item["source_term"],
            target_term=item["target_term"], preferred_target=item["target_term"], category=AUTO_CATEGORY,
            domain="GENERAL", locked=False, lock_level="SOFT",
            notes="Tự chọn trước khi dịch để thống nhất cả cuốn; sửa hoặc xoá nếu chưa hợp.",
        ))
        added += 1
    db.commit()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"source_hash": digest, "terms": chosen}, ensure_ascii=False, indent=1), encoding="utf-8")
    return added
