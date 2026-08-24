import os
from pathlib import Path
from typing import List, Dict, Any
from app.config import settings


class FontManager:
    @staticmethod
    def list_available_fonts() -> List[Dict[str, Any]]:
        """
        Discovers fonts available in fonts/ directory and common system font locations.
        """
        fonts = [
            {"name": "Noto Serif", "family": "'Noto Serif', Georgia, serif", "is_vietnamese_optimized": True, "type": "serif"},
            {"name": "Noto Sans", "family": "'Noto Sans', Arial, sans-serif", "is_vietnamese_optimized": True, "type": "sans-serif"},
            {"name": "Inter", "family": "'Inter', system-ui, sans-serif", "is_vietnamese_optimized": True, "type": "sans-serif"},
            {"name": "Times New Roman", "family": "'Times New Roman', Times, serif", "is_vietnamese_optimized": True, "type": "serif"},
            {"name": "Georgia", "family": "Georgia, serif", "is_vietnamese_optimized": True, "type": "serif"},
            {"name": "Arial", "family": "Arial, sans-serif", "is_vietnamese_optimized": True, "type": "sans-serif"},
        ]

        # Scan custom user fonts in fonts/
        if settings.FONTS_DIR.exists():
            for f in settings.FONTS_DIR.glob("*.*"):
                if f.suffix.lower() in [".ttf", ".otf", ".woff2"]:
                    fonts.append({
                        "name": f.stem,
                        "family": f"'{f.stem}', sans-serif",
                        "is_vietnamese_optimized": True,
                        "path": str(f),
                        "type": "custom"
                    })

        return fonts
