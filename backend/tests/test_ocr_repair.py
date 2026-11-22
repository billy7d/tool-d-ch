"""Đọc lại chữ OCR vỡ (chữ nghiêng, tiếng Anh), thống nhất tên riêng, khôi phục chữ hoa đầu chương."""
from collections import Counter
from types import SimpleNamespace

import pytest
from PIL import Image

from app.services.ebook import ocr_repair
from app.services.ebook.ocr_repair import (
    Word,
    compose,
    fuse,
    is_garbled,
    merge_line,
    read_back,
    restore_drop_caps,
    unify_rare_names,
)
from app.services.ebook.vi_spell import VietSyllableModel


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setattr(ocr_repair, "_english", {
        "washington", "post", "the", "review", "journal", "today", "seattle", "times", "of", "bowl", "using",
    })


def _model() -> VietSyllableModel:
    unigrams = Counter({"học": 500, "bọc": 40, "về": 900, "vế": 20, "vẻ": 60, "toán": 200, "và": 2000, "va": 50,
                        "tồi": 80, "côi": 5, "năm": 700, "nhất": 400, "mặt": 300})
    return VietSyllableModel(unigrams, Counter())


def test_is_garbled():
    for token in ("US⁄1", "J2sb7ze#øn", "toá72", "cø/", "Neumansk,cgacy"):
        assert is_garbled(token), token
    for token in ("Toảzy", "COVID-19", "7th", "10-10-10", "và/hoặc", "http://www.nfl.com/news", "(1994),"):
        assert not is_garbled(token), token


def _w(text, x0, x1, conf=0.9):
    return Word(text, (x0, 0, x1, 40), conf)


def test_merge_line_replaces_garbled_words_with_agreeing_readings():
    model = _model()
    original = [_w("e", 0, 20), _w("J2sb7ze#øn", 30, 230, 0.4), _w("Pos:", 240, 300)]
    readings = [
        ("vie", [_w("e", 0, 20), _w("J2sb7ze#øn", 30, 230, 0.3), _w("Pos:", 240, 300)]),
        ("vie", [_w("Washbingtơn", 32, 228, 0.5), _w("Post:", 240, 305)]),
        ("eng", [_w("Washington", 31, 229, 0.96), _w("Post:", 240, 305, 0.95)]),
    ]
    words, changed = merge_line(original, readings, model)
    assert [w.text for w in words] == ["e", "Washington", "Post:"]
    assert changed == [False, True, True]


def test_merge_line_fuses_italic_vietnamese_but_keeps_upright_words():
    model = _model()
    # "toá72 bọc" là "toán học" in nghiêng: "eng" thấy đúng mặt chữ "hoc", "vie" thấy dấu.
    original = [_w("toá72", 0, 100, 0.1), _w("bọc", 110, 170, 0.96), _w("va", 180, 220, 0.96)]
    readings = [
        ("vie", [_w("toá72", 0, 100, 0.1), _w("bọc", 110, 170), _w("va", 180, 220)]),
        ("vie", [_w("toán", 0, 100, 0.95), _w("bọc", 110, 170, 0.92), _w("và", 180, 220, 0.86)]),
        ("eng", [_w("todn", 0, 100, 0.9), _w("hoc", 110, 170, 0.96), _w("va", 180, 220, 0.94)]),
    ]
    words, _ = merge_line(original, readings, model)
    # "va" (chữ đứng) không bị đổi thành "và" chỉ vì bản đọc trên ảnh nắn thẳng thấy dấu huyền.
    assert [w.text for w in words] == ["toán", "học", "va"]


def test_merge_line_leaves_unknown_syllables_to_the_spell_corrector():
    model = _model()
    original = [_w("tổi", 0, 60, 0.8), _w("|", 70, 75)]
    readings = [("vie", [_w("côi", 0, 60, 0.9), _w("3", 70, 75)]), ("eng", [_w("toi", 0, 60), _w("1", 70, 75)])]
    words, changed = merge_line(original, readings, model)
    assert [w.text for w in words] == ["tổi", "|"] and not any(changed)


def test_fuse_takes_letters_from_eng_and_marks_from_vie():
    model = _model()
    assert fuse("bọc", "hoc", model) == "học"
    assert fuse("uể", "vé", model) == "về"
    assert fuse("abc", "abcd", model) is None


def test_compose_and_read_back_map_words_to_page_lines():
    page = Image.new("L", (1000, 800), 255)
    canvas, placed = compose(page, [(100, 100, 600, 140), (120, 500, 700, 545)])
    line = SimpleNamespace(
        text="USA Today",
        word_boxes=[(30 + 8, placed[1].top + 10, 30 + 80, placed[1].top + 40),
                    (30 + 100, placed[1].top + 10, 30 + 200, placed[1].top + 40)],
        word_confidences=[0.9, 0.9],
        word_alternatives=[],
    )
    words = read_back([line], placed)
    assert list(words) == [1]
    assert [w.text for w in words[1]] == ["USA", "Today"]
    # Toạ độ trên trang: cắt từ x = 120 - 8 (lề ghép), cộng lại phần dịch của ảnh ghép.
    assert words[1][0].box[0] == 120 - 8 + 8


def test_unify_rare_names():
    texts = ["Seahawks thua"] * 5 + ["đội Scahawks"] + ["Joe nói"] * 5 + ["Joel đến"] + ["Super Bowl"] * 4
    texts += ["trận Super Bow] chỉ"] + ["theo ông"] * 6 + ["Iheo ông,"]
    mapping = unify_rare_names(texts, _model())
    assert mapping["Scahawks"] == "Seahawks"
    assert mapping["Bow]"] == "Bowl"
    assert mapping["Iheo"] == "Theo"
    assert "Joel" not in mapping  # tên thật khác một chữ cái: không đổi


def test_restore_drop_caps():
    model = _model()

    def line(text, x0, y0):
        return SimpleNamespace(text=text, bbox=(x0, y0, 1500, y0 + 50), x_size=40.0, word_alternatives=[])

    lines = [line("ăm 26 tuổi, tôi nghĩ rằng", 300, 100), line("định rất rõ ràng. Tôi lớn", 300, 175),
             line("trung học tư thục danh tiếng", 200, 250)]
    page = Image.new("L", (1600, 400), 255)
    calls = []

    def recognize(image, lang, dpi):
        calls.append(image.size)
        return SimpleNamespace(text="N\n")

    assert restore_drop_caps(page, lines, recognize, 300, model) == 1
    assert lines[0].text.startswith("Năm 26")
    assert len(calls) == 1


def test_fix_number_token():
    from app.services.ebook.ocr_repair import fix_number_token

    assert fix_number_token("201ó,") == "2016,"
    assert fix_number_token("—9ó%") == "—96%"
    assert fix_number_token("có") == "có"
    assert fix_number_token("ó8") == "ó8"  # một chữ số: không đủ chắc
