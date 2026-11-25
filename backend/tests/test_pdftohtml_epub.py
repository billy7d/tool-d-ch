"""EPUB do Calibre chuyển từ PDF (pdftohtml): dựng lại đoạn, tiêu đề, chú thích như sách in.

Mẫu HTML chép theo đúng dạng của "Stumbling on Happiness" (Calibre 6.8, pdftohtml 0.36): mỗi dòng in là một <p>,
mốc trang <a id="pN">, tiêu đề mục in nghiêng hoặc mất kiểu chữ, số chú thích dính vào chữ.
"""
import zipfile

from app.models.canonical import NodeType
from app.services.extractor.epub_extractor import EPUBExtractor
from app.services.extractor.pdftohtml_epub import Line, classify, continues, convert_note_refs, join_fragments
from app.services.reconstruction.structure_builder import StructureBuilder
from app.services.translation.citation_policy import citation_passthrough, is_bibliographic_citation
from app.services.translation.quality_gate import TranslationQualityGate
from app.services.translation.worker import passthrough_text

_HEAD = (
    '<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Book</title>'
    '<meta name="generator" content="pdftohtml 0.36"/></head><body class="calibre">'
)


def _p(text: str) -> str:
    return f'<p class="calibre1">{text}</p>'


def _pdftohtml_book(tmp_path):
    body = "".join([
        _p('<a id="p1"></a><img src="index-1_1.jpg" alt="Image 1"/>'),
        _p(' <i>Stumbling  </i>'), _p(' <i>on Happiness</i>'),
        _p('<a id="p3"></a>CONTENTS'),
        _p('<a href="#p5"> <i>Acknowledgments</i></a>'),
        _p('<a href="#p9">PART I PROSPECTION</a>'),
        _p('<a href="#p9">1. Journey to Elsewhen</a>'),
        _p('<a href="#p40"> <i>Notes</i></a>'),
        _p('<a id="p5"></a>ACKNOWLEDGMENTS'),
        _p('THIS IS THE PART OF THE BOOK in which the author thanks people. It must be nice to have friends.'),
        _p('<a id="p9"></a>PART I'), _p('<b>Prospection</b>'),
        _p('CHAPTER 1'), _p(' <i>In the Blind Spot</i>'), _p(' <i>of the Mind’s Eye</i>'),
        _p('O, that a man might know'), _p('The end of this day’s business ere it come! '),
        _p('Shakespeare,  <i>Julius Caesar</i>'),
        _p('PRIESTS VOW TO REMAIN CELIBATE, and Phineas Gage was a foreman who worked with a'),
        _p('rod that the squirrel that stashes a nut in my yard'),
        _p('“knows” about the future in the same way that a falling rock'),
        _p('“knows” about gravity, as the brain does.1'),
        _p(' <i>The Joy of Next</i>'),
        _p('If you were asked to name the greatest achievement of the brain, you might name the bridge. Similarly,'),
        _p('<a id="p10"></a><img src="index-10_1.jpg" alt="Image 2"/>'),
        _p('the increase in the size of the brain grew by 3.5 percent, as Phineas noted.2'),
        _p(' <i>Fig. 2. </i> The frontal lobe.'),
        _p('Prospection and Emotion'),
        _p('The most obvious answer is that thinking about the future can be pleasurable, said Gage. 3 The next'),
        _p('sentence was about the .03'),
        _p('percent increase in weight.'),
        _p('<a id="p40"></a>NOTES'),
        _p(' <i>Chapter 1: In the Blind Spot of the Mind’s Eye</i>'),
        _p('1. W. A. Roberts, “Are Animals Stuck in Time?” <i>Psychological Bulletin</i> 128: 473–89 (2002). '),
        _p('2. M. M. Haith, “The Development of Future Thinking,” in <i>Planning</i>, ed. S. '),
        _p('L. Friedman (Mahwah, N.J.: Lawrence Erlbaum, 1997), 25–42. '),
        _p('3. Phineas apparently took the rod with him wherever he went for the rest of his life. See'),
        _p('D. Dennett, <i>Kinds of Minds</i> (New York: Basic Books, 1996).'),
    ])
    # Ghi zip trực tiếp như Calibre: ebooklib khi ghi dựng lại <head> và bỏ thẻ generator của pdftohtml.
    opf = (
        '<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0" '
        'unique-identifier="uid"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:opf="http://www.idpf.org/2007/opf"><dc:title>Stumbling on Happiness</dc:title>'
        '<dc:creator opf:role="aut">Daniel Gilbert</dc:creator><dc:language>en</dc:language>'
        '<dc:identifier id="uid">x</dc:identifier><meta name="cover" content="cover"/></metadata><manifest>'
        '<item id="cover" href="cover.jpeg" media-type="image/jpeg"/>'
        '<item id="i1" href="index-1_1.jpg" media-type="image/jpeg"/>'
        '<item id="i2" href="index-10_1.jpg" media-type="image/jpeg"/>'
        '<item id="s0" href="index_split_000.html" media-type="application/xhtml+xml"/>'
        '<item id="s1" href="index_split_001.html" media-type="application/xhtml+xml"/>'
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/></manifest>'
        '<spine toc="ncx"><itemref idref="s0"/><itemref idref="s1"/></spine></package>'
    )
    ncx = (
        '<?xml version="1.0" encoding="utf-8"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        '<head/><docTitle><text>Stumbling on Happiness</text></docTitle><navMap>'
        '<navPoint id="n1" playOrder="1"><navLabel><text>PART I</text></navLabel>'
        '<content src="index_split_000.html#p9"/></navPoint>'
        '<navPoint id="n2" playOrder="2"><navLabel><text>Notes</text></navLabel>'
        '<content src="index_split_000.html#p40"/></navPoint></navMap></ncx>'
    )
    container = (
        '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    path = tmp_path / "book.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", opf)
        archive.writestr("toc.ncx", ncx)
        archive.writestr("index_split_000.html", _HEAD + body + "</body></html>")
        archive.writestr("index_split_001.html", _HEAD + '<p class="calibre1"><h1>Document Outline</h1><ul><li>'
                         '<a href="index_split_000.html#p9">PART I</a></li></ul></p></body></html>')
        archive.writestr("cover.jpeg", b"\xff\xd8cover")
        for name in ("index-1_1.jpg", "index-10_1.jpg"):
            archive.writestr(name, b"\xff\xd8jpeg-" + name.encode())
    return path


def _nodes(tmp_path):
    extractor = EPUBExtractor(tmp_path / "project")
    data = extractor.extract_document(_pdftohtml_book(tmp_path))
    document = StructureBuilder.build_canonical_document("p", "book.epub", data["pages"], data["assets"])
    return document, data, [node for chapter in document.chapters for node in chapter.nodes]


def test_paragraph_fragments_are_joined_and_structure_rebuilt(tmp_path):
    document, data, nodes = _nodes(tmp_path)
    texts = [node.content for node in nodes]
    headings = [(node.content, node.metadata.heading_level) for node in nodes if node.type == NodeType.HEADING]

    assert data["metadata"]["title"] == "Stumbling on Happiness" and data["metadata"]["author"] == "Daniel Gilbert"
    # Trang mục lục in sẵn và trang "Document Outline" không thành nội dung cần dịch.
    assert not any("CONTENTS" in text or "Journey to Elsewhen" in text or "Document Outline" in text for text in texts)
    assert texts[0] == "Stumbling on Happiness"
    assert headings[:6] == [
        ("ACKNOWLEDGMENTS", 1), ("PART I", 1), ("Prospection", 1), ("CHAPTER 1", 2),
        ("In the Blind Spot of the Mind’s Eye", 2), ("The Joy of Next", 3),
    ]
    assert ("Prospection and Emotion", 3) in headings and ("NOTES", 1) in headings
    # Chữ in hoa mở đoạn về chữ thường, giữ tên riêng gặp ở chỗ khác trong sách.
    assert "This is the part of the book in which the author thanks people. It must be nice to have friends." in texts
    joined = next(text for text in texts if text.startswith("Priests vow"))
    assert joined == (
        "Priests vow to remain celibate, and Phineas Gage was a foreman who worked with a rod that the squirrel that "
        "stashes a nut in my yard “knows” about the future in the same way that a falling rock “knows” about gravity, "
        "as the brain does. [1]"
    )
    # Đề từ ngắn không bị dính vào nhau hay vào đoạn mở chương, và được đánh dấu trích dẫn.
    epigraph = next(node for node in nodes if node.content == "O, that a man might know")
    assert epigraph.metadata.extra["layout"] == {"quote": True}
    assert "Shakespeare, Julius Caesar" in texts
    assert [chapter.title for chapter in document.chapters][:3] == ["Introduction", "ACKNOWLEDGMENTS", "PART I"]


def test_images_follow_the_paragraph_they_interrupt_and_note_refs_are_sequential(tmp_path):
    document, data, nodes = _nodes(tmp_path)
    texts = [node.content for node in nodes]
    asset_ids = {asset["id"] for asset in data["assets"]}
    images = [node for node in nodes if node.type == NodeType.IMAGE]

    # Ảnh bìa trang 1 bỏ (EPUB có bìa riêng); ảnh giữa đoạn dời ra sau đoạn, trỏ đúng asset đã lưu.
    assert len(images) == 1 and images[0].metadata.image_asset_id in asset_ids
    position = texts.index(images[0].content)
    assert texts[position - 1].startswith("If you were asked") and texts[position - 1].endswith("as Phineas noted. [2]")
    assert "3.5 percent" in texts[position - 1]
    assert texts[position + 1] == "Fig. 2. The frontal lobe."
    assert "said Gage. [3] The next sentence was about the .03 percent increase in weight." in " ".join(texts)
    assert EPUBExtractor.cover_image(tmp_path / "book.epub") == b"\xff\xd8cover"


def test_notes_become_footnotes_and_citations_keep_their_source(tmp_path):
    _, _, nodes = _nodes(tmp_path)
    notes = [node for node in nodes if node.type == NodeType.FOOTNOTE]

    assert [note.metadata.footnote_number for note in notes] == ["1", "2", "3"]
    assert notes[1].content.endswith("ed. S. L. Friedman (Mahwah, N.J.: Lawrence Erlbaum, 1997), 25–42.")
    assert any(node.type == NodeType.HEADING and node.content.startswith("Chapter 1:") and node.metadata.heading_level == 3 for node in nodes)
    assert is_bibliographic_citation(notes[0].content) and is_bibliographic_citation(notes[1].content)
    assert not is_bibliographic_citation(notes[2].content)  # lời bình của tác giả vẫn được dịch
    assert passthrough_text("footnote", notes[0].content) == notes[0].content
    assert passthrough_text("paragraph", notes[0].content) is None
    kept = citation_passthrough("5. B. M. Hood, “Gravity Biases,” Science 2: 35–41 (1999). See also D. Washburn (1992).")
    assert kept == "5. B. M. Hood, “Gravity Biases,” Science 2: 35–41 (1999). Xem thêm D. Washburn (1992)."
    gate = TranslationQualityGate().validate(notes[0].content, notes[0].content, {})
    assert gate.passed


def test_join_rules_do_not_merge_separate_lines():
    assert continues("the squirrel that stashes a nut in my yard", "“knows” about the future")
    assert continues("we can usually estimate—", "Psychologists")
    assert continues("“Yum!” if they were hungry and “Yuck!”", "if they were full) and used")
    assert not continues("Shakespeare, Julius Caesar", "PRIESTS VOW TO REMAIN CELIBATE, physicians")
    assert not continues("O, that a man might know", "The end of this day’s business ere it come!")
    assert not continues("a good chasing.", "They include Sissela Bok")
    assert not continues("as it does.2", "But what")
    # Danh xưng viết tắt cuối trang không kết câu: "(Ms." + "Dash’s). If you think…"
    assert continues("an equally fabulous life with a few additional merely satisfactory years (Ms.", "Dash’s). If you")
    assert continues("as Dr.", "Gilbert noted")
    assert not continues("He was a good man, said Mrs. Moss.", "The next day")
    # Tên tác giả bị ngắt dòng sau chữ cái viết tắt (trang nguồn hình minh họa).
    assert continues("Figure 14 reprinted from C. Walker, “Some Variations,” in R. Chester and J.", "Peel, eds., Equalities")
    assert continues("Figure 20 reprinted with permission from T. D.", "Wilson, D. B. Centerbar, and D. T. Gilbert")
    assert not continues("We called it Plan B.", "The next morning we left.")
    # Dấu chấm lửng cách quãng bị ngắt trang giữa câu.
    assert continues("(“The boss will march into my office, shut the door behind him .", ". .”) without also imagining")


def test_note_refs_ignore_years_decimals_and_out_of_order_numbers():
    lines = [Line("text", "NOTES"), Line("text", "1. A note (1999).")]
    blocks, level = classify([Line("text", "CHAPTER 1"), Line("text", "In 1848, about 3.5 tons fell.3 Then it fell.1 Later.2"),
                              *lines], ["Notes"])
    blocks = join_fragments(blocks)
    assert convert_note_refs(blocks, level) == 2
    assert blocks[1].text == "In 1848, about 3.5 tons fell.3 Then it fell. [1] Later. [2]"
    # Số chú thích dính sau một con số ("for $300.23 Because") chỉ được nhận khi đúng số đang chờ.
    blocks, level = classify([Line("text", "CHAPTER 1"), Line("text", "It cost $300.1 Because of that, 4.2 percent paid."),
                              *lines], ["Notes"])
    blocks = join_fragments(blocks)
    assert convert_note_refs(blocks, level) == 1
    assert blocks[1].text == "It cost $300. [1] Because of that, 4.2 percent paid."
    # Hai mục chú thích in liền nhau trên một dòng được tách đúng số.
    blocks, _ = classify([Line("text", "NOTES"), Line("text", "21. J. Schooler, “Verbal,” Cog. 22: 36–71 (1990.) 22. G. McConkie, “Is,” P. 2 (1980).")],
                         ["Notes"])
    assert [(block.note_number, block.text[:6]) for block in blocks if block.kind == "note"] == [("21", "21. J."), ("22", "22. G.")]


def test_gate_accepts_part_labels_species_names_and_loanwords():
    gate = TranslationQualityGate()
    # Báo nhầm gặp ở 65 đoạn đầu của sách: "PART I" -> "PHẦN I" bị coi là thêm tham chiếu.
    assert gate.validate("PART I", "PHẦN I", {}).passed
    assert not gate.validate("Part 2 explains it.", "Chương này giải thích.", {}).passed
    assert gate.validate("the part I liked", "phần tôi thích", {}).passed
    assert gate.validate(
        "the giant sea slug known as Aplysia parvula can learn",
        "loài sên biển khổng lồ có tên Aplysia parvula có thể học", {},
    ).passed
    assert gate.validate("a T-bone with extra cholesterol", "một miếng T-bone kèm thêm cholesterol", {}).passed
    assert not gate.validate("The dog ran.", "The dog chạy.", {}).passed
    assert gate.validate("• In Part II, “Subjectivity,” I will tell you", "• Trong Phần II, “Tính chủ quan”, tôi sẽ kể", {}).passed


def test_gate_rejects_prompt_instructions_echoed_into_the_translation():
    gate = TranslationQualityGate()
    source = "20. As far as I can tell, the word prefeel was first used as a song title on the 1999 album Prize."
    echoed = (
        "20.\n\nNhắc lại: chỉ dùng chữ Quốc ngữ, không dùng chữ Hán, dịch hết mọi từ tiếng Anh (trừ tên riêng, mã, "
        "đường dẫn). Theo tôi được biết, từ prefeel lần đầu được dùng làm tên một bài hát trong album Prize năm 1999."
    )
    result = gate.validate(source, echoed, {})
    assert not result.passed and "PROMPT_ECHO" in {issue["code"] for issue in result.issues}
    clean = "20. Theo tôi được biết, từ prefeel lần đầu được dùng làm tên một bài hát trong album Prize năm 1999."
    assert "PROMPT_ECHO" not in {issue["code"] for issue in gate.validate(source, clean, {}).issues}


def test_caption_lines_split_after_an_initial_are_rejoined():
    from app.services.extractor.pdftohtml_epub import Block

    blocks = [
        Block("caption", "Figure 14 reprinted from C. Walker, “Some Variations,” in R. Chester and J.", page=1),
        Block("text", "Peel, eds., Equalities and Inequalities in Family Life (London: Academic Press, 1977), 127–39.", page=1),
        Block("text", "Copyright © 1977. Reprinted by permission of Elsevier.", page=1),
        Block("caption", "Fig. 5. Cows standing in a field near the barn at dusk", page=2),
        Block("text", "The next paragraph of the chapter starts here and runs on for a while.", page=2),
    ]
    texts = [block.text for block in join_fragments(blocks)]
    assert texts[0].endswith("in R. Chester and J. Peel, eds., Equalities and Inequalities in Family Life (London: Academic Press, 1977), 127–39.")
    assert texts[1] == "Copyright © 1977. Reprinted by permission of Elsevier."
    # Thân bài ngay sau một chú thích hình thiếu dấu chấm vẫn là đoạn riêng.
    assert texts[2:] == ["Fig. 5. Cows standing in a field near the barn at dusk", "The next paragraph of the chapter starts here and runs on for a while."]


def test_saved_document_keeps_image_assets_for_export(tmp_path):
    """Lưu tài liệu phải ghi cả bản ghi ảnh: thiếu nó, EPUB xuất ra chỉ có bìa (sách thật rơi mất 25 hình)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db.models import AssetModel, Base
    from app.db.repository import StructureRepository

    document, data, nodes = _nodes(tmp_path)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repository = StructureRepository(db)
    repository.save_canonical_document("book", document)
    repository.save_canonical_document("book", repository.get_canonical_document("book"))  # lưu lại không nhân đôi

    saved = {asset.id: asset.original_path for asset in db.query(AssetModel).all()}
    image_ids = {node.metadata.image_asset_id for node in nodes if node.type == NodeType.IMAGE}
    assert image_ids and image_ids <= set(saved)
    reloaded = {asset.id for asset in repository.get_canonical_document("book").assets}
    assert image_ids <= reloaded


def test_note_refs_become_numbered_superscripts():
    from app.services.ebook.book_plan import _note_refs

    marked = _note_refs("ngượng ngùng và hối tiếc. [1] Rắc rối là [12], còn [3]")
    assert marked == "ngượng ngùng và hối tiếc.<sup>1</sup> Rắc rối là<sup>12</sup>, còn<sup>3</sup>"
    assert all(ord(char) >= 32 for char in marked)
