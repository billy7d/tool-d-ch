"""Gate không coi tên thư viện, khóa link markdown và tên riêng cạnh URL là chữ tiếng Anh chưa dịch.

Dịch thử README của các thư viện Python bằng Gemma 4: phần lớn đoạn đúng bị chặn vì "pydantic", "lxml", "libxslt",
"[contrib]", "Kozea (https://kozea.fr/)". Chữ chưa dịch thật ("strong", "cathect" trong sách văn học) vẫn phải bị bắt.
"""
import pytest

from app.services.translation import quality_gate
from app.services.translation.glossary_presets import preset_entries
from app.services.translation.quality_gate import TranslationQualityGate

ENGLISH = {"strong", "library", "libraries", "water", "binding", "parser", "specification", "the", "use", "here"}


@pytest.fixture(autouse=True)
def small_dictionary(monkeypatch):
    monkeypatch.setattr(quality_gate, "_english_common_word", lambda word: word.lower() in ENGLISH)
    monkeypatch.setattr(quality_gate, "_DICTIONARY_PROBE", None)


def residue(source, target, terms=None):
    issues = TranslationQualityGate().validate(source, target, terms if terms is not None else {}).issues
    return [issue["message"] for issue in issues if issue["code"] == "ENGLISH_RESIDUE"]


def test_lone_lowercase_package_heading_is_kept_as_a_name():
    assert residue("pydantic", "pydantic") == []
    assert residue("requests", "requests") == []


def test_markdown_reference_key_is_kept_but_untranslated_link_text_is_still_caught():
    assert residue(
        "See our [detailed contributing documentation][contrib] for many ways to contribute.",
        "Xem [tài liệu đóng góp chi tiết][contrib] để biết nhiều cách đóng góp.",
    ) == []
    assert residue(
        "The [ASGI specification][asgi] fills this gap.",
        "[ASGI specification][asgi] lấp đầy khoảng trống này.",
    )


def test_capitalised_name_whose_lowercase_form_is_only_in_a_url_is_a_name():
    assert residue(
        "WeasyPrint has been created and developed by Kozea (https://kozea.fr/).",
        "WeasyPrint được Kozea (https://kozea.fr/) tạo ra và phát triển.",
    ) == []


def test_link_text_left_in_english_is_still_caught_when_the_url_path_repeats_it():
    assert residue(
        "For guidance, see [Contributing to Pydantic](https://docs.pydantic.dev/contributing/).",
        "Để được hướng dẫn, hãy xem [Contributing to Pydantic](https://docs.pydantic.dev/contributing/).",
    )


def test_unknown_lowercase_words_in_a_technical_source_are_names_but_english_words_are_not():
    source = "lxml is a Pythonic, mature binding for the libxml2 and libxslt libraries."
    assert residue(source, "lxml là lớp liên kết cho các thư viện libxml2 và libxslt.") == []
    assert residue(source, "lxml là lớp liên kết strong cho các thư viện libxml2 và libxslt.")


def test_unknown_words_in_literary_prose_are_still_untranslated_residue():
    assert residue(
        "anything we cathect, without regard for the quality of that relationship",
        "mọi thứ chúng ta cathect, không quan tâm đến chất lượng của mối quan hệ đó",
    )


def test_a_technical_document_treats_every_paragraph_as_technical():
    technical = preset_entries(["TECHNICAL"])
    assert residue("Install pytest first.", "Hãy cài pytest trước.", technical) == []
    assert residue("Install pytest first.", "Hãy cài pytest trước.")
    assert residue("Use the parser here.", "Dùng parser ở đây.", technical)


def test_measurement_prefixes_and_programming_language_names_are_not_residue():
    assert residue("Detectors run in sub-millisecond time.", "Các bộ phát hiện chạy dưới mili giây.") == []
    technical = preset_entries(["TECHNICAL"])
    assert residue(
        "Notifications are handled by the notify rust library.",
        "Các thông báo được xử lý bởi thư viện Rust notify.", technical,
    ) == []
    assert residue("A python is a large snake.", "Python là một loài rắn lớn.")


def test_without_a_dictionary_unknown_words_are_not_exempted(monkeypatch):
    monkeypatch.setattr(quality_gate, "_english_common_word", lambda word: False)
    monkeypatch.setattr(quality_gate, "_DICTIONARY_PROBE", None)
    assert residue(
        "lxml is a Pythonic, mature binding for the libxml2 and libxslt libraries.",
        "lxml là lớp liên kết cho các thư viện libxml2 và libxslt.",
    )
