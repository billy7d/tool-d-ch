import pytest
from app.services.reconstruction.paragraph_repair import ParagraphRepair


def test_hyphen_repair():
    # Regular hyphenated wrap should be joined
    broken_text = "The invest-\nment strategy proved effective."
    repaired = ParagraphRepair.repair_hyphenation(broken_text)
    assert "investment strategy" in repaired

    # Legitimate compound should be preserved
    compound_text = "They used state-of-\nthe-art tools."
    repaired_compound = ParagraphRepair.repair_hyphenation(compound_text)
    assert "state-of-the-art" in repaired_compound


def test_line_joining():
    lines = [
        "The market continued to expand",
        "during the next decade despite",
        "several downturns."
    ]
    joined = ParagraphRepair.join_broken_lines(lines)
    assert joined == "The market continued to expand during the next decade despite several downturns."
