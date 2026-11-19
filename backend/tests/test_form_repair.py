from app.services.translation.form_repair import repair_form, repair_quotes, split_stuck_words


def test_splits_vietnamese_word_stuck_to_a_source_word():
    source = "Their belief in God and what Rachel would do."
    assert split_stuck_words(source, "Niềm tin vàoGod và một ngày kiaRachel sẽ làm.") == "Niềm tin vào God và một ngày kia Rachel sẽ làm."


def test_keeps_brand_names_and_tokens_present_in_the_source():
    assert split_stuck_words("He bought an iPhone on eBay.", "Anh mua một chiếc iPhone trên eBay.") == "Anh mua một chiếc iPhone trên eBay."
    assert split_stuck_words("Use the newValue field.", "Dùng trường newValue.") == "Dùng trường newValue."
    # Chữ viết hoa không có trong nguồn thì không đủ căn cứ để tách.
    assert split_stuck_words("They went home.", "Họ vềNha.") == "Họ vềNha."


def test_missing_space_after_sentence_period():
    assert split_stuck_words("How to resolve it? Then Ted spoke.", "Cách giải quyết.Ted lên tiếng.") == "Cách giải quyết. Ted lên tiếng."
    assert split_stuck_words("See example.Com and Ph.D. holders, etc.", "Xem example.Com và người có bằng Ph.D., v.v. Tiếp theo.") == (
        "Xem example.Com và người có bằng Ph.D., v.v. Tiếp theo."
    )


def test_stray_period_before_a_source_word():
    assert split_stuck_words("a moment of relief and joy", "khoảnh khắc của sự.relief và vui sướng") == "khoảnh khắc của sự relief và vui sướng"
    assert split_stuck_words("visit site.org today", "hãy vào site.org hôm nay") == "hãy vào site.org hôm nay"


def test_quotes_follow_the_source_when_counts_match():
    source = "“Kathy,” I went on, “I can think of reasons.”"
    assert repair_quotes(source, "”Kathy“, tôi nói tiếp: ”Tôi nghĩ ra nhiều lý do.\"") == "“Kathy”, tôi nói tiếp: “Tôi nghĩ ra nhiều lý do.”"
    assert repair_quotes("“Stay,” he said.", "\"Ở lại,\" anh nói.") == "“Ở lại,” anh nói."
    assert repair_quotes("“Stay,” he said.", "“ Ở lại, ” anh nói.") == "“ Ở lại, ” anh nói."


def test_quotes_left_alone_without_a_clear_basis():
    # Số ngoặc khác nguồn, nguồn dùng ngoặc thẳng, hoặc có dấu hiệu code.
    assert repair_quotes("“Stay,” he said.", "Anh bảo \"ở lại\" rồi \"đi\".") == "Anh bảo \"ở lại\" rồi \"đi\"."
    assert repair_quotes('He said "stay".', "Anh nói ”ở lại“.") == "Anh nói ”ở lại“."
    assert repair_quotes("Call “run” now.", 'Gọi print("run") ngay.') == 'Gọi print("run") ngay.'


def test_appends_missing_final_closing_quote():
    source = "He said, “I will stay until you are ready.”"
    assert repair_quotes(source, "Ông nói: “Tôi sẽ ở lại đến khi bạn sẵn sàng.") == "Ông nói: “Tôi sẽ ở lại đến khi bạn sẵn sàng.”"
    # Nguồn không kết thúc bằng ngoặc đóng thì không biết ngoặc thiếu nằm ở đâu.
    source_mid = "He said, “I will stay,” and left the room."
    assert repair_quotes(source_mid, "Ông nói: “Tôi sẽ ở lại, rồi rời phòng.") == "Ông nói: “Tôi sẽ ở lại, rồi rời phòng."


def test_repair_form_is_a_no_op_on_clean_text():
    source = "“Stay,” he said. God is here."
    clean = "“Ở lại,” anh nói. Chúa ở đây."
    assert repair_form(source, clean) == clean
    assert repair_form(source, "") == ""


def test_gate_rejects_punctuation_garbage_but_not_normal_punctuation():
    from app.services.translation.quality_gate import TranslationQualityGate

    gate = TranslationQualityGate()
    source = "When I was a child I was beaten, and nobody ever came to help me at all in those years."
    bad = gate.validate(source, "Khi còn nhỏ - bị đánh đập,, ——..", {})
    assert "PUNCTUATION_GARBAGE" in [issue["code"] for issue in bad.issues] and not bad.passed
    good = gate.validate(source, "Khi còn nhỏ tôi bị đánh đập, v.v... và chẳng ai đến giúp tôi trong những năm đó.", {})
    assert "PUNCTUATION_GARBAGE" not in [issue["code"] for issue in good.issues]
    # Nguồn có sẵn chuỗi đó thì bản dịch giữ lại là đúng.
    kept = gate.validate("She wrote “” on the board and waited for the class to answer her.", "Cô viết “” lên bảng và chờ cả lớp trả lời.", {})
    assert "PUNCTUATION_GARBAGE" not in [issue["code"] for issue in kept.issues]
