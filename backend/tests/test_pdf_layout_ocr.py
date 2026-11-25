"""Bố cục sách scan sau OCR: chú thích vắt trang, văn xuôi canh trái, mục lục in sẵn và tiêu đề chương."""
from collections import Counter

import pytest

from app.services.ebook import pdf_layout
from app.services.ebook.pdf_layout import Line, PageLayout, PdfLayoutAnalyzer, Span, _Para, _Style, _Unit
from app.services.ebook.vi_spell import VietSyllableModel


def _analyzer() -> PdfLayoutAnalyzer:
    analyzer = PdfLayoutAnalyzer.__new__(PdfLayoutAnalyzer)
    analyzer.body_size = 12.0
    analyzer.pitch = 16.0
    analyzer.vocab = set()
    analyzer.hyphen_vocab = set()
    analyzer.heading_fixes = 0
    return analyzer


def _unit(text, page=1, y0=100.0, size=12.0, lines=1, **flags) -> _Unit:
    para = _Para()
    para._push(text, frozenset())
    unit = _Unit(kind="text", page=page, bbox=[50.0, y0, 380.0, y0 + 14 * lines], segment=(page, 0), para=para,
                 size=size, body=abs(size - 12.0) <= 0.22 * 12.0, ocr=True, line_count=lines,
                 first_y0=y0, last_y0=y0 + 14 * (lines - 1), last_y1=y0 + 14 * lines)
    for key, value in flags.items():
        setattr(unit, key, value)
    return unit


def _line(text, x0, x1, y) -> Line:
    return Line([Span(text, 10.0)], x0, y, x1, y + 12)


def test_ragged_prose_is_not_verse():
    analyzer = _analyzer()
    prose = [
        _line("Với 26 giây cuối cùng trong trận chung kết", 30, 226, 100),
        _line("Super Bowl XLIX, huấn luyện viên đội", 30, 205, 114),
        _line("Seahawks Peter Carroll đã đưa ra một trong", 30, 222, 128),
        _line("những quyết định bị nhiều người chỉ trích nhất", 30, 228, 142),
        _line("trong lịch sử bóng bầu dục.", 30, 150, 156),
    ]
    assert analyzer._ragged_prose(prose)
    poem = [
        _line("Trăm năm trong cõi người ta,", 120, 260, 100),
        _line("Chữ tài chữ mệnh khéo là ghét nhau.", 100, 290, 114),
        _line("Trải qua một cuộc bể dâu,", 120, 250, 128),
        _line("Những điều trông thấy mà đau đớn lòng.", 100, 300, 142),
    ]
    assert not analyzer._ragged_prose(poem)


def test_footnote_continuations_are_marked_and_joined():
    analyzer = _analyzer()
    page = PageLayout(number=5, width=435, height=648)
    body = _unit("Dĩ nhiên, nghiêng không chỉ xuất hiện trong poker, hoặc một lời", page=5, y0=470, lines=5,
                 last_full=True)
    note = _unit("* Pinball là trò chơi bắn những viên kim loại nhỏ vào các cọc đặt trên", page=5, y0=581,
                 size=9.6, footnote=True)
    rest = _unit("mặt bàn nghiêng (ND).", page=5, y0=594, size=10.3)
    units = [body, note, rest]
    analyzer._mark_footnote_continuations(page, units)
    assert rest.footnote
    # Phần còn lại của chú thích trang trước, chữ nhỏ ở cuối trang sau, mở đầu bằng chữ thường.
    page6 = PageLayout(number=6, width=435, height=648)
    body6 = _unit("Hành vi đó ai cũng có, đó là bản chất con người.", page=6, y0=100, lines=20)
    carry = _unit("vị về việc sự thiên lệch trong quyết định bị phơi bày", page=6, y0=476, size=9.6, lines=8)
    analyzer._mark_footnote_continuations(page6, [body6, carry])
    assert carry.footnote
    joined = analyzer._join_units([body, note, rest], _Style(indent=True))
    notes = [u for u in joined if u.footnote]
    assert len(notes) == 1 and notes[0].para.visible().endswith("đặt trên mặt bàn nghiêng (ND).")


def test_toc_cleanup_and_chapter_title_repair(monkeypatch):
    model = VietSyllableModel(Counter({"lời": 500, "cảm": 800, "ơn": 300}), Counter({("lời", "cảm"): 40}))
    monkeypatch.setattr(VietSyllableModel, "shared", classmethod(lambda cls: model))
    analyzer = _analyzer()
    units = [
        _unit("| MỤC LŨ", page=8, y0=60, size=32, heading_like=True),
        _unit("CHƯƠNG 1 | Cuộc đời là một ván poker, không phải một ván cờ vua 7", page=8, y0=100, toc_entry=True),
        _unit("Pete Carroll và những gã thầy bói nói dựa Sự nguy hiểm của resulting 10", page=8, y0=116,
              toc_entry=True),
        _unit("Nhanh hay chết: não chúng ta không được lập trình l4", page=8, y0=132, toc_entry=True),
        _unit("“Một trăm cái bánh hamburger... và một ly sữa lắc", page=8, y0=148),
        _unit("sô cô la lớn”: cách trách nhiệm giải trình cải thiện việc ra quyết định 160", page=8, y0=164,
              toc_entry=True),
        _unit("LỒI CẢM ƠN 273", page=8, y0=180, toc_entry=True, size=15.7, heading_like=True),
        _unit("CHƯƠNG 1", page=17, y0=128, size=23.7, heading_like=True),
        _unit("Cuộc đời là một ván poker, không phủ", page=17, y0=165, size=23.0, lines=2),
        _unit("\" ˆ LẠ", page=17, y0=195, size=6.1),
        _unit("¡ mộồt ván cờ vud", page=17, y0=196, size=20.7, heading_like=True, align="center"),
        _unit("Pete Carroll và những gã thầy bói nói dựa", page=17, y0=307, size=17.0, heading_like=True),
        _unit("Sự nguy hiểm của resulting", page=18, y0=100, size=17.0, heading_like=True),
        _unit("Nhanh hay chết: não chúng ta không được lập trình", page=19, y0=100, size=17.0, heading_like=True),
        _unit("| LỜI CẢM 0N", page=300, y0=100, size=24.0, heading_like=True),
    ]
    out = analyzer._repair_toc_and_headings(units)
    texts = [u.para.visible() for u in out]
    assert texts[:7] == [
        "MỤC LỤC",
        "CHƯƠNG 1 | Cuộc đời là một ván poker, không phải một ván cờ vua",
        "Pete Carroll và những gã thầy bói nói dựa",
        "Sự nguy hiểm của resulting",
        "Nhanh hay chết: não chúng ta không được lập trình",
        "“Một trăm cái bánh hamburger... và một ly sữa lắc sô cô la lớn”: cách trách nhiệm giải trình cải thiện "
        "việc ra quyết định",
        "LỜI CẢM ƠN",
    ]
    assert texts[7:10] == ["CHƯƠNG 1", "Cuộc đời là một ván poker, không phải một ván cờ vua",
                           "Pete Carroll và những gã thầy bói nói dựa"]
    assert texts[-1] == "LỜI CẢM ƠN"
    # Dòng mục lục không được nhận nhầm thành tiêu đề chương.
    assert all(u.size <= 12.0 and not u.heading_like for u in out[1:7])


def test_replace_tokens_keeps_word_data_aligned():
    span = Span("Journal øƒ Law,", 12.0, word_confidences=[0.9, 0.3, 0.8],
                word_boxes=[(0, 0, 90, 30), (100, 0, 130, 30), (140, 0, 200, 30)],
                word_alternatives=[(), ((50.0, ()),) * 2, ()])
    positions = PdfLayoutAnalyzer._replace_tokens(span, {1: ["of", "the"], 2: []})
    assert span.text == "Journal of the"
    assert positions == [1, 2]
    assert span.word_confidences == [0.9, 0.9, 0.9]
    assert span.word_boxes == [(0, 0, 90, 30), (100, 0, 115, 30), (115, 0, 130, 30)]
    assert span.word_alternatives == [(), (), ()]


def test_verify_summary_reports_pages_to_check():
    analyzer = _analyzer()
    analyzer.verify_stats = Counter({"flagged": 570, "quick": 50, "fixed": 280, "guessed": 4, "confirmed": 160,
                                     "kept": 20, "removed": 9, "dropped": 1, "passes": 4})
    analyzer.unreadable = [(17, "ˆ", "removed", ""), (314, "Š", "guessed", "S"), (308, "dir. 2ø", "unresolved", "")]
    note = analyzer._verify_summary()
    assert "570 chữ không đạt" in note and "đã sửa 334 chữ" in note and "bỏ 10 cụm" in note
    assert note.endswith("còn 2 chữ chưa chắc chắn ở trang 308, 314.")
    analyzer.verify_stats = Counter()
    assert analyzer._verify_summary() == ""


def test_consistent_collocations_follow_the_majority():
    spans = []
    for _ in range(4):
        spans.append((Span("tránh những cái bẫy quyết định", 12.0), "tránh những cái bãy quyết định"))
    odd = Span("sa vào những cái bay nằm giữa", 12.0)
    spans.append((odd, "sa vào những cái bãy nằm giữa"))
    keep = Span("con chim bay qua", 12.0)
    spans.append((keep, "con chim bay qua"))
    assert PdfLayoutAnalyzer._consistent_collocations(spans) == 1
    assert odd.text == "sa vào những cái bẫy nằm giữa"
    assert keep.text == "con chim bay qua"
