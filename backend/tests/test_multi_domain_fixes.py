"""Các lỗi tìm ra khi dịch thử 7 sách miễn phí thuộc 7 lĩnh vực (truyện, sinh học, kinh tế, chính trị, triết học,
kinh doanh, lập trình) qua đúng đường nhập và dịch của app."""
from ebooklib import epub

from app.models.canonical import NodeType
from app.services.extractor.epub_extractor import EPUBExtractor
from app.services.reconstruction.header_footer_filter import HeaderFooterFilter
from app.services.reconstruction.structure_builder import StructureBuilder
from app.services.translation.form_repair import drop_double_titles, repair_form, wrap_lost_quotes
from app.services.translation.glossary_presets import preset_entries
from app.services.translation.numeric_validator import NumericValidator
from app.services.translation.quality_gate import TranslationQualityGate, _malformed_words, _non_vietnamese_words


def _book(tmp_path):
    book = epub.EpubBook()
    book.set_identifier("test")
    book.set_title("Test Book")
    book.set_language("en")
    # Tên file đặt ngược thứ tự đọc để kiểm việc đọc theo spine.
    second = epub.EpubHtml(title="Second Story", file_name="a-second.xhtml", lang="en")
    second.content = (
        "<html><head><title>Second Story</title></head><body>"
        "<h2>Second Story</h2><p>“Yes.”</p><p>“Yes.”</p><p>“Yes.”</p><p>It ended.</p></body></html>"
    )
    first = epub.EpubHtml(title="VII: Objections", file_name="b-first.xhtml", lang="en")
    first.content = (
        "<html><head><title>VII: Objections</title></head><body>"
        "<hgroup><h2>VII</h2><p epub:type=\"title\">Objections</p></hgroup>"
        "<p>Run <code>git apply</code> first, at twenty percent.<a href=\"#n1\" epub:type=\"noteref\">697</a></p>"
        "<ul><li>You can clone a repository, or</li><li><p>use a paragraph item.</p></li></ul>"
        "<pre><code class=\"language-console\">$ git apply --check fix.patch\n$ git am</code></pre>"
        "<h4>Applying a Patch with am</h4>"
        "<figure><img src=\"x.png\"/><figcaption>Figure 8. The lifecycle of files</figcaption></figure>"
        "<dl><dt>Cirripedes</dt><dd>An order of Crustaceans.</dd></dl>"
        "<table><tr><td>Years</td><td>Barrels</td></tr></table>"
        "<p>On two great laws" + chr(0xFEFF) + "—Unity of Type.</p>"
        "</body></html>"
    )
    for item in (second, first):
        book.add_item(item)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [first, second]
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)
    return path


def test_epub_import_keeps_all_content_in_reading_order(tmp_path):
    data = EPUBExtractor(tmp_path / "project").extract_document(_book(tmp_path))
    document = StructureBuilder.build_canonical_document("p", "book.epub", data["pages"], data["assets"])
    nodes = [node for chapter in document.chapters for node in chapter.nodes]
    texts = [node.content for node in nodes]

    assert [chapter.title for chapter in document.chapters][-2:] == ["VII: Objections", "Second Story"]
    assert "Run `git apply` first, at twenty percent. [697]" in texts
    assert "You can clone a repository, or" in texts and "use a paragraph item." in texts
    code = next(node for node in nodes if node.type == NodeType.CODE_BLOCK)
    assert code.content == "$ git apply --check fix.patch\n$ git am" and code.metadata.code_language == "console"
    assert "Figure 8. The lifecycle of files" in texts
    assert "Cirripedes" in texts and "An order of Crustaceans." in texts
    assert "Years | Barrels" in texts
    assert "On two great laws—Unity of Type." in texts
    # Đoạn ngắn lặp lại không còn bị coi là đầu trang và bị xoá.
    assert texts.count("“Yes.”") == 3
    headings = {node.content: node.metadata.heading_level for node in nodes if node.type == NodeType.HEADING}
    assert headings["Applying a Patch with am"] == 4 and headings["Objections"] == 3


def test_blocks_without_coordinates_are_never_headers_or_footers():
    pages = [{"page_number": index, "blocks": [{"type": "text", "text": "“Yes.”"}]} for index in range(5)]
    repeated = HeaderFooterFilter.identify_repeated_headers_footers(pages)
    assert repeated == set()
    assert HeaderFooterFilter.filter_blocks(pages[0]["blocks"], {"“Yes.”"}) == pages[0]["blocks"]


def test_numbers_written_as_words_months_and_ordinals_are_not_additions():
    assert NumericValidator.validate("dated the 27th of July, 1694.", "ngày 27 tháng 7 năm 1694.").passed
    assert NumericValidator.validate(
        "tallies had been at forty, and fifty, and sixty percent discount", "giấy nợ bị giảm giá 40%, 50% và 60%",
    ).passed
    assert NumericValidator.validate("In pursuance of the 7th Anne, c. vii", "Theo năm thứ bảy của Anne, c. vii").passed
    assert NumericValidator.validate("In November of his eighth year", "Vào tháng 11 năm 8 tuổi").passed
    # Số thật sự bị thêm hoặc bị mất vẫn bị bắt.
    assert not NumericValidator.validate("dated the 27th of July, 1694.", "ngày 27 tháng 8 năm 1694.").passed
    assert not NumericValidator.validate("In 1708 the bank lent £400,000.", "Năm 1708 ngân hàng cho vay.").passed


def test_vietnamese_word_fused_with_a_foreign_word_is_caught():
    assert _malformed_words("", "các thuật ngữ như Chúa, Thần hay Đếextrême mang lại") == ["Đếextrême"]
    assert _malformed_words("", "Giêsu, Kitô, Mátxcơva, Oasinhtơn, Nghiêng ngả") == []


def test_title_case_heading_words_left_in_english_are_residue():
    source = "Miscellaneous Objections to the Theory of Natural Selection"
    assert "Objection" in _non_vietnamese_words(source, "Các Objection Miễn cưỡng đối với Lý thuyết Chọn lọc Tự nhiên")
    assert _non_vietnamese_words("Wealth of Nations by Adam Smith", "Sự giàu có của các quốc gia của Adam Smith") == []
    # Trích dẫn tên sách vẫn được giữ tên gốc.
    assert _non_vietnamese_words("* The Prophet (New York: Knopf, 1951), pp. 17-18.", "* The Prophet (New York: Knopf, 1951), trang 17-18.") == []


def test_short_source_that_grows_into_paragraphs_is_rejected():
    gate = TranslationQualityGate()
    bad = gate.validate("Entropy and Original Sin", "Entropy và Nguyên tội\n\nEntropy là một khái niệm trong vật lý học mô tả sự giảm dần.", {})
    assert "SHORT_SOURCE_EXPANSION" in [issue["code"] for issue in bad.issues] and not bad.passed
    assert "SHORT_SOURCE_EXPANSION" in [i["code"] for i in gate.validate("Packfiles", "Packfiles là tập tin chứa dữ liệu Git được nén.", {}).issues]
    good = gate.validate("Persevere", "Kiên trì", {})
    assert "SHORT_SOURCE_EXPANSION" not in [issue["code"] for issue in good.issues]


def test_preset_terms_and_source_code_words_are_not_residue():
    presets = preset_entries(["TECHNICAL"])
    source = "If someone pushes a change, run `git svn rebase` to fetch it before you commit."
    target = "Nếu ai đó push một thay đổi, hãy chạy git svn rebase để fetch nó trước khi commit."
    assert _non_vietnamese_words(source, target, presets) == []
    # Chữ tiếng Anh thật sự bị sót vẫn bị bắt.
    assert _non_vietnamese_words(source, target + " Bạn phải either làm vậy.", presets) == ["either"]


def test_double_english_and_vietnamese_titles_are_merged():
    assert drop_double_titles("“Now then, Mr. Cocksure,” said the salesman", "“Bây giờ thì ông Mr. Cocksure,” người bán hàng nói") == (
        "“Bây giờ thì ông Cocksure,” người bán hàng nói"
    )
    assert drop_double_titles("“Mrs. Oakshott, 117, Brixton Road,” read Holmes.", "“Bà Mrs. Oakshott, 117 Brixton Road,” Holmes đọc.") == (
        "“Bà Oakshott, 117 Brixton Road,” Holmes đọc."
    )
    # Nguồn không có danh xưng tiếng Anh thì không đụng tới.
    assert drop_double_titles("He met Dr Watson.", "Ông gặp bác sĩ Dr Watson.") == "Ông gặp bác sĩ Watson."
    assert drop_double_titles("He met Watson.", "Ông gặp ông Mr Watson.") == "Ông gặp ông Mr Watson."


def test_fully_quoted_paragraph_gets_its_quotes_back():
    source = "“That’s the list of the folk from whom I buy. Just read it out to me.”"
    assert wrap_lost_quotes(source, "Đây là danh sách những người tôi mua hàng. Hãy đọc nó cho tôi nghe.") == (
        "“Đây là danh sách những người tôi mua hàng. Hãy đọc nó cho tôi nghe.”"
    )
    assert wrap_lost_quotes("He said, “Read it.”", "Ông nói hãy đọc nó.") == "Ông nói hãy đọc nó."
    assert repair_form(source, "“Đây là danh sách.”") == "“Đây là danh sách.”"


def test_final_commit_gate_uses_the_engine_terms():
    """Gate lúc lưu phải dùng cùng bộ thuật ngữ với gate lúc dịch (gồm preset), không thì đoạn đạt bị chặn khi lưu."""
    from types import SimpleNamespace

    import pytest
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db.models import Base, NodeModel, ProjectModel
    from app.services.translation.translation_commit import TranslationCommitService

    db = sessionmaker(bind=create_engine("sqlite:///:memory:"))()
    Base.metadata.create_all(db.get_bind())
    db.add(ProjectModel(id="p", title="P"))
    node = NodeModel(id="n", project_id="p", content="Run this before you commit your work.", node_type="paragraph",
                     status="PENDING", order_index=0, version=1)
    db.add(node)
    db.commit()
    text = "Hãy chạy lệnh này trước khi commit công việc của bạn."
    signature = SimpleNamespace(style_hash="s", glossary_hash="g", prompt_version="v")
    with pytest.raises(ValueError):
        TranslationCommitService.commit_validated_translation(db, "p", node, text, signature, {}, "mock", "test")
    db.rollback()
    TranslationCommitService.commit_validated_translation(
        db, "p", node, text, signature, {}, "mock", "test", validation_terms=preset_entries(["TECHNICAL"]),
    )
    assert node.translated_content == text
