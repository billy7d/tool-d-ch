from app.db.engine import get_project_db
from app.db.models import ChapterModel, NodeModel, ProjectModel
from app.models.canonical import DocumentNode, NodeType, SourceMapping
from app.services.reconstruction.page_break_joiner import (
    Fragment, continues_sentence, find_joins, join_chapter_nodes, join_pending_nodes, join_text,
)
from app.services.reconstruction.structure_builder import StructureBuilder

BODY = "Yet when one is dedicated to the truth this pain"
REST = "seems relatively unimportant—and less and less important."


def _node(node_id, text, page, node_type=NodeType.PARAGRAPH):
    return DocumentNode(
        id=node_id, type=node_type, content=text,
        source_mapping=SourceMapping(source_page_start=page, source_page_end=page, source_block_ids=[page * 10]),
    )


def test_lowercase_start_after_unfinished_sentence_continues():
    assert continues_sentence(BODY, REST)
    assert continues_sentence("and this also is typical", "—made not the slightest effort.")


def test_finished_sentence_or_new_sentence_does_not_continue():
    assert not continues_sentence("This pain is real.", "seems odd but it is a new block.")
    assert not continues_sentence("It grows.”", "and then more.")
    assert not continues_sentence("the source of their power.*", "see Peter Brent.")
    assert not continues_sentence("Published by Simon & Schuster", "To my parents, Elizabeth and David.")
    assert not continues_sentence("no longer at ease here, in the old dispensation,", "With an alien people.")


def test_opening_quote_continues_only_after_long_body_text():
    long_body = "x" * 220 + " lying on the couch and gently stroked her head, murmuring,"
    assert continues_sentence(long_body, "“Poor Rachel. Poor Rachel.” The gesture failed.")
    assert not continues_sentence("A note on method", "“Truth is the daughter of time.”")
    assert not continues_sentence(long_body[:-1] + ". “", "“Well, you know,” Ted replied.")


def test_join_text_keeps_hyphen_of_split_compound():
    assert join_text("not only that self-", "discipline is usually love") == "not only that self-discipline is usually love"
    assert join_text(BODY + " ", " " + REST) == f"{BODY} {REST}"


def test_same_page_and_non_adjacent_pages_are_not_joined():
    same_page = [Fragment("paragraph", BODY, 3, 3), Fragment("paragraph", REST, 3, 3)]
    skipped_page = [Fragment("paragraph", BODY, 3, 3), Fragment("paragraph", REST, 5, 5)]
    assert find_joins(same_page) == []
    assert find_joins(skipped_page) == []


def test_heading_between_halves_blocks_join_but_footnote_on_same_page_does_not():
    heading = [Fragment("paragraph", BODY, 3, 3), Fragment("heading", "Love Defined", 4, 4), Fragment("paragraph", REST, 4, 4)]
    footnote = [Fragment("paragraph", BODY, 3, 3), Fragment("footnote", "* See Brent.", 3, 3), Fragment("paragraph", REST, 4, 4)]
    assert find_joins(heading) == []
    assert find_joins(footnote) == [(0, 2)]


def test_paragraph_spanning_three_pages_joins_into_first():
    fragments = [
        Fragment("paragraph", "It began with", 1, 1), Fragment("paragraph", "a long middle that never", 2, 2),
        Fragment("paragraph", "ends until here.", 3, 3), Fragment("paragraph", "but this is separate.", 4, 4),
    ]
    assert find_joins(fragments) == [(0, 1), (0, 2)]


def test_join_chapter_nodes_merges_content_and_source_mapping():
    nodes = [_node("p1", BODY, 35), _node("p2", REST, 36), _node("p3", "A new paragraph.", 36)]
    joined = join_chapter_nodes(nodes)
    assert [node.id for node in joined] == ["p1", "p3"]
    assert joined[0].content == f"{BODY} {REST}"
    assert (joined[0].source_mapping.source_page_start, joined[0].source_mapping.source_page_end) == (35, 36)
    assert joined[0].source_mapping.source_block_ids == [350, 360]


def test_structure_builder_joins_paragraph_split_by_page_break():
    def block(block_id, text):
        return {"type": "text", "block_id": block_id, "text": text, "bbox": [70, 100, 500, 200], "font_size": 11.0}

    pages = [
        {"page_number": 1, "blocks": [block(1, "A complete opening paragraph."), block(2, BODY)]},
        {"page_number": 2, "blocks": [block(3, REST), block(4, "Another paragraph follows here.")]},
    ]
    document = StructureBuilder.build_canonical_document("p", "book.pdf", pages, [])
    contents = [node.content for chapter in document.chapters for node in chapter.nodes]
    assert contents == ["A complete opening paragraph.", f"{BODY} {REST}", "Another paragraph follows here."]
    assert document.metadata.total_nodes == 3


def test_join_pending_nodes_leaves_translated_halves_alone():
    project_id = "join-pending"
    db = get_project_db(project_id)
    try:
        db.add(ProjectModel(id=project_id, title="Book"))
        db.add(ChapterModel(id="c1", project_id=project_id, title="One", order_index=0))
        rows = [
            ("n1", BODY, 1, "PENDING", None), ("n2", REST, 2, "PENDING", None),
            ("n3", BODY, 2, "TRANSLATED", "Bản dịch"), ("n4", REST, 3, "PENDING", None),
            ("n5", BODY, 3, "PENDING", None), ("n6", REST, 0, "PENDING", None),
        ]
        for index, (node_id, text, page, status, translation) in enumerate(rows):
            mapping = {"source_page_start": page, "source_page_end": page, "source_block_ids": [index]} if page else {}
            db.add(NodeModel(id=node_id, project_id=project_id, chapter_id="c1", node_type="paragraph", content=text,
                             translated_content=translation, status=status, order_index=index, source_mapping=mapping))
        db.commit()

        assert join_pending_nodes(db, project_id) == 1
        db.commit()
        remaining = db.query(NodeModel).order_by(NodeModel.order_index).all()
        assert [row.id for row in remaining] == ["n1", "n3", "n4", "n5", "n6"]
        assert remaining[0].content == f"{BODY} {REST}"
        assert remaining[0].source_mapping["source_page_end"] == 2
        assert remaining[0].source_mapping["source_block_ids"] == [0, 1]
        assert join_pending_nodes(db, project_id) == 0
    finally:
        db.close()
