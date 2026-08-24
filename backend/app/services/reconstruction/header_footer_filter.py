import re
from collections import Counter
from typing import List, Dict, Any, Set


class HeaderFooterFilter:
    @staticmethod
    def identify_repeated_headers_footers(pages_data: List[Dict[str, Any]]) -> Set[str]:
        """
        Scans all pages and identifies text snippets that appear repeatedly near top/bottom margins.
        """
        top_texts = []
        bottom_texts = []
        page_numbers_patterns = re.compile(r"^\d{1,4}$|^page\s*\d{1,4}$|^\d{1,4}\s*\|\s*.*$", re.IGNORECASE)

        total_pages = len(pages_data)
        if total_pages < 2:
            return set()

        for page in pages_data:
            page_height = page.get("height", 842.0)
            blocks = page.get("blocks", [])
            
            for b in blocks:
                if b.get("type") != "text":
                    continue
                text = b.get("text", "").strip()
                if not text:
                    continue

                bbox = b.get("bbox", [0, 0, 0, 0])
                y0, y1 = bbox[1], bbox[3]

                # Top 12% of page
                if y1 < page_height * 0.12:
                    top_texts.append(text)
                # Bottom 12% of page
                elif y0 > page_height * 0.88:
                    bottom_texts.append(text)

        top_counts = Counter(top_texts)
        bottom_counts = Counter(bottom_texts)

        # Repetition threshold: appears in > 15% of pages or at least 3 times
        threshold = max(3, int(total_pages * 0.15))
        repeated = set()

        for text, count in top_counts.items():
            if count >= threshold or page_numbers_patterns.match(text):
                repeated.add(text)

        for text, count in bottom_counts.items():
            if count >= threshold or page_numbers_patterns.match(text):
                repeated.add(text)

        return repeated

    @staticmethod
    def filter_blocks(blocks: List[Dict[str, Any]], repeated_headers_footers: Set[str], page_height: float = 842.0) -> List[Dict[str, Any]]:
        """
        Filters out confirmed repeated headers, footers and standalone page numbers.
        """
        filtered = []
        standalone_num_pattern = re.compile(r"^\d{1,4}$|^page\s*\d{1,4}$", re.IGNORECASE)

        for b in blocks:
            if b.get("type") != "text":
                filtered.append(b)
                continue

            text = b.get("text", "").strip()
            bbox = b.get("bbox", [0, 0, 0, 0])
            y0, y1 = bbox[1], bbox[3]
            is_margin = (y1 < page_height * 0.12) or (y0 > page_height * 0.88)

            # Check if repeated header/footer in margins
            if text in repeated_headers_footers and is_margin:
                continue

            # Check if isolated page number in top/bottom margin
            if is_margin and standalone_num_pattern.match(text):
                continue

            filtered.append(b)

        return filtered
