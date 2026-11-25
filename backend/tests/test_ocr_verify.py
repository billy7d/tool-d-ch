"""Điều kiện bắt buộc cho từng chữ OCR và vòng quét lại vùng chữ lỗi."""
from collections import Counter
from types import SimpleNamespace

import pytest
from PIL import Image

from app.services.ebook import ocr_repair, ocr_verify as ov
from app.services.ebook.ocr_repair import shear, unshear_x
from app.services.ebook.vi_spell import VietSyllableModel

ENGLISH = {
    "of", "the", "journal", "times", "new", "law", "economic", "general", "with", "and", "gid", "final", "mill",
    "review", "in", "fame", "human", "is", "it", "drives", "a", "was", "street", "wall", "pdf", "can", "don",
}


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setattr(ocr_repair, "_english", set(ENGLISH))
    ov._known_cache.clear()
    yield
    ov._known_cache.clear()


def _model() -> VietSyllableModel:
    unigrams = Counter({
        "tồi": 80, "tổi": 1, "về": 900, "vé": 300, "vê": 60, "giá": 700, "trị": 600, "giữa": 500, "mái": 120,
        "không": 3000, "thoải": 90, "một": 4000, "chương": 300, "trình": 400, "cuộc": 800, "phỏng": 90,
        "vấn": 700, "cả": 900, "và": 5000, "tư": 900, "của": 4000, "cược": 200, "những": 2000, "học": 900,
        "tiền": 600, "trở": 500, "lý": 400, "thuyết": 200, "nói": 700, "đặt": 600, "ta": 3000, "chúng": 2000,
        "nhất": 900, "tệ": 120, "ván": 80, "cờ": 200, "vua": 300, "xèng": 1,
    })
    bigrams = Counter({("trở", "về"): 50, ("giữa", "giá"): 5, ("giá", "trị"): 80, ("mái", "giữa"): 2,
                       ("chúng", "ta"): 900})
    return VietSyllableModel(unigrams, bigrams)


# ---- Điều kiện bắt buộc ---------------------------------------------------------------------------------

@pytest.mark.parametrize("token", [
    "Journal", "of", "và", "tồi,", "2016", "6,25%", "(91,44cm).", "109–26.", "2/2/2015,", "978-604-1-16374-4.",
    "https://www.nfl.com/news", "why_pete_carroll_s", "COVID-19", "60th", "S&P", "#MAGA", "P.7,", "F.3d",
    "—>", "|", "(b)", "Seahawks", "NEJM", "don’t", "x",
])
def test_valid_tokens_pass(token):
    assert ov.check_token(token, _model()) is None, token


@pytest.mark.parametrize("token, reason", [
    ("øƒ", "garbled"), ("7e", "garbled"), ("J2z72l", "garbled"), ("`", "symbol"), ("‹««", "symbol"),
    ("z", "stray"), ("ofFame", "joined"), ("máigiữa", "joined"), ("JournAl", "case"), ("tổi", "unknown"),
    ("jWz", "unknown"), ("C2/cb¿mg", "garbled"),
])
def test_errors_are_classified(token, reason):
    assert ov.check_token(token, _model()) == reason


def test_book_vocabulary_accepts_rare_real_words():
    assert ov.check_token("xèng", _model()) == "unknown"
    assert ov.check_token("xèng", _model(), accepted={"xèng"}) is None


@pytest.mark.parametrize("token, fixed", [
    ("ofFame", "of Fame"), ("máigiữa", "mái giữa"), ("cuộcphỏngvấn", "cuộc phỏng vấn"),
    ("mộtchươngtrình", "một chương trình"), ("109~26.", "109–26."), ("1ó,", "16,"), ("cả,một", "cả, một"),
    ("tư:vấn", "tư vấn"), ("JournAl", "Journal"), ("MilI", "Mill"), ("don”t", "don’t"), ("ñnal.pdf.", "final.pdf."),
    ("htrp://www.x.com/a⁄b", "http://www.x.com/a/b"), ("Ba2017):", "Ba 2017):"),
])
def test_quick_fixes(token, fixed):
    assert ov.quick_fix(token, _model()) == fixed


@pytest.mark.parametrize("token", ["Orson", "Andrew", "McFly", "6,25%", "øƒ", "7e"])
def test_quick_fix_leaves_names_and_unreadable_words(token):
    assert ov.quick_fix(token, _model()) is None


def test_caps_heading_confusions():
    model = _model()
    assert ov.fix_caps_word("tƯỢt", model) == "CƯỢC"
    assert ov.fix_caps_word("HỌt", model) == "HỌC"
    assert ov.fix_caps_word("Seattle", model) is None


def test_join_split_words():
    tokens = ["Wall", "Street", "Jour", "nal,", "26/6/2016,"]
    reasons = {3: "unknown"}
    monkey = set(ENGLISH) | {"journal"}
    ocr_repair._english = monkey
    assert ov.join_split_words(tokens, reasons, _model()) == {2: ["Journal,"], 3: []}


# ---- Nắn chữ nghiêng ----------------------------------------------------------------------------------------

def test_shear_keeps_the_whole_line_and_maps_back():
    """Bản cũ cắt mất mép trái của các dòng phía trên ảnh ghép cao: mọi điểm ảnh phải còn trên ảnh đã nắn."""
    image = Image.new("L", (200, 400), 255)
    for x, y in ((0, 0), (5, 10), (150, 390)):
        image.putpixel((x, y), 0)
    k = 0.25
    sheared = shear(image, k)
    dark = [(x, y) for y in range(sheared.height) for x in range(sheared.width) if sheared.getpixel((x, y)) < 128]
    for x, y in ((0, 0), (5, 10), (150, 390)):
        assert any(abs(unshear_x(dx, dy, k) - x) <= 1 and dy == y for dx, dy in dark), (x, y)


# ---- Bỏ phiếu ----------------------------------------------------------------------------------------------

def _region(tokens, reasons, readings, before="", after=""):
    region = ov.Region(0, len(tokens) - 1, list(tokens), list(reasons), 0, 100, before=before, after=after)
    region.readings = list(readings)
    return region


def test_garbled_word_fixed_by_agreeing_readings():
    region = _region(["øƒ"], ["garbled"], [("vie", "oƒ"), ("eng", "of"), ("vie", "of"), ("eng", "of")],
                     before="Journal", after="Law,")
    ov.decide(region, _model(), Counter())
    assert (region.status, region.result) == ("fixed", "of")


def test_vietnamese_marks_come_from_the_vie_model():
    # "eng" đọc "về" thành "vé"; dấu lấy theo bản "vie" ghép với mặt chữ của "eng".
    region = _region(["0ể"], ["garbled"], [("vie", "0ể"), ("eng", "vé"), ("vie", "0ê"), ("vie+eng", "vé"),
                                             ("eng", "vé"), ("vie", "uề")], before="Trở", after="tương")
    ov.decide(region, _model(), Counter(), final=True)
    assert region.result == "về"
    region = _region(["tổi"], ["unknown"], [("vie", "tồi"), ("eng", "toi"), ("vie", "tồi")], before="nhất")
    ov.decide(region, _model(), Counter())
    assert (region.status, region.result) == ("fixed", "tồi")


def test_vietnamese_context_rejects_english_only_words():
    readings = [("vie", "giá"), ("eng", "gid"), ("vie", "giá"), ("eng", "gid"), ("vie", "giá"), ("eng", "gid")]
    region = _region(["ø¿á"], ["garbled"], readings, before="giữa", after="trị")
    ov.decide(region, _model(), Counter())
    assert (region.status, region.result) == ("fixed", "giá")


def test_italic_letter_confusion_fixed_from_eng_letters():
    # Chữ nghiêng: "vie" đọc "h" thành "b" ổn định ("bạnh"), "eng" thấy đúng mặt chữ "hanh"; ghép ra "hạnh".
    readings = [("vie", "bạnh"), ("eng", "hanh"), ("vie", "bạnh"), ("eng", "hanh"), ("vie", "bạnh")]
    model = _model()
    model.unigrams["hạnh"] = 300
    region = _region(["bạnh"], ["unknown"], readings, before="thuyết", after="phúc)")
    ov.decide(region, model, Counter())
    assert (region.status, region.result) == ("fixed", "hạnh")


def test_same_letters_with_other_marks_do_not_confirm():
    # "tiền", "tiến" có cùng mặt chữ với "tiển" nhưng không khẳng định được "tiển".
    region = _region(["tiển"], ["unknown"], [("vie", "tiền"), ("eng", "tien"), ("vie", "tiến")])
    ov.decide(region, _model(), Counter(), final=True)
    assert region.status == "kept"


def test_rare_word_confirmed_when_readings_agree():
    region = _region(["premortem"], ["unknown"], [("vie", "premortem"), ("eng", "premortem"), ("vie", "premortem")])
    ov.decide(region, _model(), Counter())
    assert region.status == "confirmed"


def test_unknown_name_needs_the_book_or_many_votes():
    readings = [("eng", "Azach"), ("vie", "Azach"), ("eng", "Azach")]
    region = _region(["dzách"], ["unknown"], readings)
    ov.decide(region, _model(), Counter(), final=True)
    assert region.status == "kept"  # chữ gốc hình dạng bình thường: không đổi sang một tên đoán mò
    region = _region(["J]im"], ["symbol"], [("eng", "Jim"), ("vie", "Jim"), ("eng", "Jim"), ("vie+eng", "Jim")])
    ov.decide(region, _model(), Counter())
    assert region.result == "Jim"


def test_noise_is_dropped_and_unreadable_text_reported():
    region = _region(["‹««"], ["symbol"], [("vie", ""), ("eng", ""), ("vie", "")])
    ov.decide(region, _model(), Counter())
    assert (region.status, region.result) == ("dropped", "")
    # Còn mảnh chữ có thật: không xoá, báo lại để soát.
    region = _region(["Conspiracies—Hou¿"], ["garbled"], [("vie", ""), ("eng", "\x02"), ("vie", "")])
    ov.decide(region, _model(), Counter({"conspiracies": 2}), final=True)
    assert region.status == "unresolved"


def test_final_guess_needs_both_models():
    one_model = [("eng", "Zinberg,"), ("eng", "Zinberg,"), ("eng", "Zinberg,"), ("vie", "Z7anberg,")]
    region = _region(["Z¡inberg,"], ["garbled"], one_model)
    ov.decide(region, _model(), Counter(), final=True)
    assert region.status != "guessed"
    both = [("eng", "Zinberg,"), ("vie", "Zinberg,"), ("eng", "Zinberg,"), ("vie", "Z7anberg,")]
    region = _region(["Z¡inberg,"], ["garbled"], both)
    ov.decide(region, _model(), Counter(), final=True)
    assert region.result == "Zinberg,"


def test_duplicated_neighbour_is_not_repeated():
    model = _model()
    region = _region(["¿bể"], ["garbled"], [], before="có", after="sẽ")
    assert ov.adapt_candidate(region, "thể sẽ", model, Counter({"thể": 3})) in (None, "thể")


# ---- Vòng quét lại ----------------------------------------------------------------------------------------

def _fake_recognizer(answers):
    """Trả về một chữ cho mỗi dòng của ảnh ghép, đặt đúng chỗ vùng lỗi (ảnh không nắn, tỷ lệ 1)."""
    calls = []

    def recognize(image, lang, dpi):
        calls.append(lang)
        lines = []
        for top, (x0, x1, text) in answers:
            lines.append(SimpleNamespace(text=text, word_boxes=[(x0 + ov._GAP, top + 10, x1 + ov._GAP, top + 30)]))
        return SimpleNamespace(lines=lines)

    return recognize, calls


def test_rescan_stops_when_every_region_is_resolved():
    model = _model()
    page = Image.new("L", (600, 200), 255)
    job = ov.line_job(page, (50, 50, 400, 90), 300, ["Journal", "øƒ", "Law,"],
                      [(60, 55, 150, 85), (160, 55, 190, 85), (200, 55, 260, 85)], {1: "garbled"})
    region = job.regions[0]
    _, tops = ov.compose_lines([job.image])
    recognize, calls = _fake_recognizer([(tops[0], (region.x0, region.x1, "of"))])
    passes = ((ov.Strategy("eng"), ov.Strategy("vie"), ov.Strategy("eng")),
              (ov.Strategy("vie"), ov.Strategy("eng"), ov.Strategy("vie")))
    jobs, outcome = ov.rescan(iter([job]), recognize, model, Counter(), passes=passes)
    assert outcome == {"fixed": 1}
    assert region.result == "of"
    assert len(calls) == 3  # lượt hai không cần chạy


def test_region_text_reports_merged_readings():
    region = ov.Region(0, 0, ["1ó,"], ["garbled"], 100, 140)
    assert ov.region_text([("1466976", 0, 95), ("16,", 102, 138)], region, 40) == "16,"
    assert ov.region_text([("146697616,", 0, 138)], region, 40) == ov.MERGED
    assert ov.region_text([("x", 400, 420)], region, 40) == ""
