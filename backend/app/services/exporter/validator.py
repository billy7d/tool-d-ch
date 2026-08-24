from pathlib import Path
from typing import Tuple


class ExportValidator:
    @staticmethod
    def validate_export_file(file_path: Path, expected_format: str) -> Tuple[bool, str]:
        """
        Validates generated export artifact according to PRD Section 129.
        """
        if not file_path.exists():
            return False, "File xuất bản không tồn tại trên ổ đĩa."

        size_bytes = file_path.stat().st_size
        if size_bytes <= 0:
            return False, "File xuất bản có dung lượng 0 bytes (bị rỗng)."

        # Format-specific validation
        if expected_format.lower() == "pdf":
            try:
                with open(file_path, "rb") as f:
                    header = f.read(5)
                    if not header.startswith(b"%PDF-"):
                        return False, "File PDF không hợp lệ (thiếu header %PDF-)."
            except Exception as e:
                return False, f"Không thể đọc file PDF: {e}"

        elif expected_format.lower() == "epub":
            try:
                with open(file_path, "rb") as f:
                    header = f.read(4)
                    if not header.startswith(b"PK\x03\x04"):
                        return False, "File EPUB không đúng chuẩn ZIP container."
            except Exception as e:
                return False, f"Không thể đọc file EPUB: {e}"

        return True, "Hợp lệ"
