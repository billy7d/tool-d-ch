"""Dọn trang "Mục lục" in sẵn của sách scan và dùng nó để sửa tiêu đề."""
from collections import Counter

from app.services.ebook.toc_repair import (
    chapter_number,
    clean_entry,
    is_page_ref,
    merge_readings,
    parse_entry,
    split_merged,
    strip_page_ref,
)
from app.services.ebook.vi_spell import VietSyllableModel


def test_page_refs_including_ocr_misreads():
    for token in ("45", "l4", "ó8", "b»)", "55)"):
        assert is_page_ref(token), token
    for token in ("2008", "|", "poker", "và"):
        assert not is_page_ref(token), token
    assert strip_page_ref("Nhanh hay chết: lý trí l4") == "Nhanh hay chết: lý trí"
    assert strip_page_ref("Cảnh báo hai phút") == "Cảnh báo hai phút"


def test_chapter_entries_keep_their_number():
    assert chapter_number("(HƯƠNG 5 | Bất đồng ý kiến") == 5
    assert chapter_number("| (HƯƠNG 2 | Muốn cược không?") == 2
    assert clean_entry("CHƯƠNG 1 | Cuộc đời là một ván poker 7") == "CHƯƠNG 1 | Cuộc đời là một ván poker"
    assert clean_entry("(HƯƠNG 5 | Bất đồng ý kiến để chiến thắng 181") == "CHƯƠNG 5 | Bất đồng ý kiến để chiến thắng"
    entry = parse_entry("CHƯƠNG 2 | Muốn cược không? 45")
    assert (entry.label, entry.title, entry.chapter) == ("CHƯƠNG 2", "Muốn cược không?", 2)


def test_split_entries_merged_on_one_line():
    assert split_merged("Hầu hết là cược với chính mình b») Sự đặt cược của chúng ta 57") == [
        "Hầu hết là cược với chính mình", "Sự đặt cược của chúng ta 57"]
    assert split_merged("CHƯƠNG 1 | Cuộc đời") == ["CHƯƠNG 1 | Cuộc đời"]


def test_merge_readings_picks_real_words():
    model = VietSyllableModel(Counter({"lời": 500, "lồi": 20, "cảm": 800, "ơn": 300, "lối": 400, "lõi": 30, "tư": 900}),
                              Counter({("lời", "cảm"): 40}))
    assert merge_readings("LỒI CẢM ƠN", "LỜI CẢM 0N", model) == "LỜI CẢM ƠN"
    assert merge_readings("Lõi tư duy", "Lối tư duy", model) == "Lối tư duy"
    assert merge_readings("Một hai", "Một hai ba", model) == "Một hai"  # khác số chữ: giữ bản chính
