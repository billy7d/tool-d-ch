import pytest
from app.services.reconstruction.header_footer_filter import HeaderFooterFilter


def test_repeated_header_filter():
    pages_data = [
        {
            "page_number": i,
            "height": 842.0,
            "blocks": [
                {"type": "text", "text": "THE INTELLIGENT INVESTOR", "bbox": [50, 20, 300, 40]},
                {"type": "text", "text": f"Chapter content on page {i}", "bbox": [50, 100, 400, 600]},
                {"type": "text", "text": f"{i}", "bbox": [250, 800, 300, 820]}
            ]
        }
        for i in range(1, 10)
    ]

    repeated = HeaderFooterFilter.identify_repeated_headers_footers(pages_data)
    assert "THE INTELLIGENT INVESTOR" in repeated

    # Test filtering on page 1
    filtered = HeaderFooterFilter.filter_blocks(pages_data[0]["blocks"], repeated, page_height=842.0)
    texts = [b["text"] for b in filtered]
    assert "THE INTELLIGENT INVESTOR" not in texts
    assert "Chapter content on page 1" in texts
