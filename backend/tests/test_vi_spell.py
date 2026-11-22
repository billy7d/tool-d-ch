"""Sửa lỗi dấu tiếng Việt sau OCR: chỉ sửa khi ngữ cảnh rõ, không đụng chữ đúng, tên riêng, từ viết tắt."""
from collections import Counter

from app.services.ebook.vi_spell import (
    OcrSpellCorrector,
    VietSyllableModel,
    lowercase_stray_capitals,
    mark_distance,
    skeleton,
)


def test_mark_distance_counts_vietnamese_edits():
    assert skeleton("Đường") == "duong"
    assert mark_distance("tổi", "tồi") == 1  # đổi thanh
    assert mark_distance("bãy", "bẫy") == 1  # thêm dấu mũ
    assert mark_distance("dây", "đây") == 1  # d -> đ
    assert mark_distance("rằng", "răng") == 1  # mất thanh
    assert mark_distance("hóa", "hoá") == 0  # chỉ khác chỗ đặt dấu thanh
    assert mark_distance("khỏe", "khoẻ") == 0


def _model() -> VietSyllableModel:
    text = (
        "trước đây tôi đã nói như vậy . cái bẫy của sự tự tin . sợi dây điện bị đứt . "
        "kết quả tồi nhất . trước đây và bây giờ . cái bẫy rất khó thấy . dây điện . "
        "hóa học . tôi không chắc chắn . tôi không chắc"
    ) * 5
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    for sentence in text.split("."):
        tokens = sentence.split()
        unigrams.update(tokens)
        bigrams.update(zip(tokens, tokens[1:]))
    return VietSyllableModel(unigrams, bigrams)


def test_corrector_fixes_diacritics_from_context_only():
    model = _model()
    corrector = OcrSpellCorrector(model, ["Trước đây tôi đã đi qua.", "Sợi dây điện ở đó."])
    assert corrector.correct_tokens("Trước dây tôi đã nói như vậy.") == "Trước đây tôi đã nói như vậy."
    assert corrector.correct_tokens("Đó là cái bãy của sự tự tin.") == "Đó là cái bẫy của sự tự tin."
    assert corrector.correct_tokens("Kết quả tổi nhất.") == "Kết quả tồi nhất."
    # Chữ đúng theo ngữ cảnh thì giữ nguyên.
    assert corrector.correct_tokens("Sợi dây điện bị đứt.") == "Sợi dây điện bị đứt."
    # Không đổi qua lại giữa hai cách đặt dấu thanh.
    assert corrector.correct_tokens("Môn hoá học.") == "Môn hoá học."
    # "T" hoa đầu câu bị đọc thành "l".
    assert corrector.correct_tokens("“lôi không chắc chắn.”") == "“Tôi không chắc chắn.”"
    # Giữ nguyên khoảng trắng, xuống dòng, dấu câu.
    assert corrector.correct_tokens("trước  dây\ntôi, đã") == "trước  đây\ntôi, đã"
    assert sum(corrector.changes.values()) >= 4


def test_corrector_leaves_names_and_foreign_words():
    corrector = OcrSpellCorrector(_model(), [])
    assert corrector.correct_tokens("Ông Pete Carroll gặp Dây ở Seattle.") == "Ông Pete Carroll gặp Dây ở Seattle."
    assert corrector.correct_tokens("resulting poker") == "resulting poker"


def test_lowercase_stray_capitals():
    model = VietSyllableModel(Counter({"vui": 500}), Counter())
    assert lowercase_stray_capitals("thuốc có VỊ đắng và", model) == "thuốc có vị đắng và"
    assert lowercase_stray_capitals("họ thấy VUI vẻ lắm", model) == "họ thấy vui vẻ lắm"
    assert lowercase_stray_capitals("giải đấu NFL năm đó", model) == "giải đấu NFL năm đó"
    assert lowercase_stray_capitals("CHƯƠNG MỘT TƯ DUY", model) == "CHƯƠNG MỘT TƯ DUY"


def _image_model() -> VietSyllableModel:
    text = (
        "kiếm tiền . tiền bạc . kiếm tiền nhiều . tôi chưa đi . tôi chưa từng . tiển . "
        "để làm gì . đề nghị này . đề nghị của tôi . vấn đề lớn . chi tiết nhỏ . các chi tiết"
    ) * 6
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    for sentence in text.split("."):
        tokens = sentence.split()
        unigrams.update(tokens)
        bigrams.update(zip(tokens, tokens[1:]))
    unigrams["tiển"] = 0
    return VietSyllableModel(unigrams, bigrams)


def test_corrector_uses_tesseract_alternatives():
    corrector = OcrSpellCorrector(_image_model(), ["Tôi cần kiếm tiền."])
    # "tiển": Tesseract đã cân nhắc "ề" (64 điểm) bên cạnh "ể" (94 điểm).
    alternatives = [
        ((95.0, ()), (95.0, ()), (95.0, ()), (95.0, ())),
        ((95.0, ()), (94.0, ()), (94.0, (("ề", 64.0),)), (95.0, ()), (95.0, ())),
    ]
    fixed = corrector.correct_tokens("kiếm tiển.", confidences=[0.96, 0.96], alternatives=alternatives)
    assert fixed == "kiếm tiền."


def test_corrector_sentence_start_lowercase_l_is_capital_t():
    corrector = OcrSpellCorrector(_image_model(), [])
    assert corrector.correct_tokens("trước. lồi chưa đi.") == "trước. Tôi chưa đi."


def test_corrector_skips_links():
    corrector = OcrSpellCorrector(_image_model(), [])
    assert corrector.correct_tokens("xem htp://www.afsa.org/ và tiền") == "xem htp://www.afsa.org/ và tiền"
