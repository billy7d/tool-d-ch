from enum import Enum
from typing import Dict, Any


class PageType(str, Enum):
    TEXT_PAGE = "TEXT_PAGE"
    SCANNED_PAGE = "SCANNED_PAGE"
    MIXED_PAGE = "MIXED_PAGE"


class ScanDetector:
    @staticmethod
    def classify_page(
        text_char_count: int,
        word_count: int,
        image_count: int,
        image_area_ratio: float,
        page_area: float
    ) -> PageType:
        """
        Classifies a PDF page into TEXT_PAGE, SCANNED_PAGE, or MIXED_PAGE.
        - SCANNED_PAGE: Little or no selectable text, but contains full or large page images.
        - TEXT_PAGE: Abundant selectable text, little or no full-page image background.
        - MIXED_PAGE: Substantial selectable text alongside significant illustrations/diagrams.
        """
        # If very few characters (< 50) and large image coverage (> 50% of page), it's scanned
        if text_char_count < 60:
            if image_count > 0 and image_area_ratio > 0.4:
                return PageType.SCANNED_PAGE
            elif image_count > 0:
                return PageType.SCANNED_PAGE
            else:
                # Blank or decorative page
                return PageType.TEXT_PAGE

        # If substantial text (> 150 chars) and low image coverage
        if text_char_count > 200 and image_area_ratio < 0.3:
            return PageType.TEXT_PAGE

        # If substantial text and substantial image
        if text_char_count >= 60 and image_count > 0 and image_area_ratio >= 0.3:
            return PageType.MIXED_PAGE

        return PageType.TEXT_PAGE
