from typing import List, Dict, Any, Tuple, Optional


class TableParser:
    @staticmethod
    def detect_table_structure(blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Attempts to detect and reconstruct simple grid tables from aligned text blocks.
        If table is too complex or ambiguous, flags for TABLE_AS_IMAGE fallback.
        """
        if len(blocks) < 4:
            return None

        # Sort blocks vertically, then horizontally
        sorted_blocks = sorted(blocks, key=lambda b: (round(b.get("bbox", [0, 0, 0, 0])[1] / 15.0), b.get("bbox", [0, 0, 0, 0])[0]))
        
        # Group into rows based on y-coordinates
        rows = []
        current_row = []
        current_y = None

        for b in sorted_blocks:
            bbox = b.get("bbox", [0, 0, 0, 0])
            y_mid = (bbox[1] + bbox[3]) / 2.0
            
            if current_y is None or abs(y_mid - current_y) < 12.0:
                current_row.append(b.get("text", "").strip())
                current_y = y_mid if current_y is None else (current_y + y_mid) / 2.0
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [b.get("text", "").strip()]
                current_y = y_mid

        if current_row:
            rows.append(current_row)

        if len(rows) < 2:
            return None

        # Check column count consistency
        col_counts = [len(r) for r in rows]
        max_cols = max(col_counts)
        min_cols = min(col_counts)

        # High confidence if column counts match
        if max_cols == min_cols and max_cols >= 2:
            return {
                "type": "table",
                "table_as_image": False,
                "confidence": 0.95,
                "rows": rows,
            }
        elif max_cols >= 2 and (max_cols - min_cols) <= 1 and len(rows) >= 3:
            # Pad uneven rows
            padded_rows = [r + [""] * (max_cols - len(r)) for r in rows]
            return {
                "type": "table",
                "table_as_image": False,
                "confidence": 0.75,
                "rows": padded_rows,
            }
        else:
            # Low confidence -> Fallback to Table as Image
            return {
                "type": "table",
                "table_as_image": True,
                "confidence": 0.40,
                "rows": rows,
                "warning": "Bảng này được giữ dưới dạng ảnh vì cấu trúc quá phức tạp để tái tạo an toàn."
            }
