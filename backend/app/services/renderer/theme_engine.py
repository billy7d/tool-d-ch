from typing import Dict, Any
from app.db.models import LayoutProfileModel


class ThemeEngine:
    PRESETS = {
        "Classic Book": {
            "page_size": "A5",
            "page_width_mm": 148.0,
            "page_height_mm": 210.0,
            "margin_top_mm": 20.0,
            "margin_bottom_mm": 20.0,
            "margin_left_mm": 20.0,
            "margin_right_mm": 20.0,
            "body_font": "'Noto Serif', serif",
            "heading_font": "'Noto Serif', serif",
            "body_font_size_pt": 11.0,
            "line_height": 1.5,
            "paragraph_spacing_pt": 3.0,
            "first_line_indent_mm": 5.0,
            "text_alignment": "justify",
            "chapter_break_mode": "next_page",
        },
        "Modern Book": {
            "page_size": "A5",
            "page_width_mm": 148.0,
            "page_height_mm": 210.0,
            "margin_top_mm": 22.0,
            "margin_bottom_mm": 22.0,
            "margin_left_mm": 18.0,
            "margin_right_mm": 18.0,
            "body_font": "'Noto Sans', sans-serif",
            "heading_font": "'Noto Sans', sans-serif",
            "body_font_size_pt": 10.5,
            "line_height": 1.55,
            "paragraph_spacing_pt": 6.0,
            "first_line_indent_mm": 0.0,
            "text_alignment": "left",
            "chapter_break_mode": "next_page",
        },
        "Academic": {
            "page_size": "A4",
            "page_width_mm": 210.0,
            "page_height_mm": 297.0,
            "margin_top_mm": 25.0,
            "margin_bottom_mm": 25.0,
            "margin_left_mm": 25.0,
            "margin_right_mm": 25.0,
            "body_font": "'Noto Serif', serif",
            "heading_font": "'Noto Serif', serif",
            "body_font_size_pt": 10.0,
            "line_height": 1.4,
            "paragraph_spacing_pt": 4.0,
            "first_line_indent_mm": 6.0,
            "text_alignment": "justify",
            "chapter_break_mode": "next_page",
        },
        "Technical Manual": {
            "page_size": "A4",
            "page_width_mm": 210.0,
            "page_height_mm": 297.0,
            "margin_top_mm": 20.0,
            "margin_bottom_mm": 20.0,
            "margin_left_mm": 20.0,
            "margin_right_mm": 20.0,
            "body_font": "'Noto Sans', sans-serif",
            "heading_font": "'Noto Sans', sans-serif",
            "body_font_size_pt": 10.5,
            "line_height": 1.5,
            "paragraph_spacing_pt": 5.0,
            "first_line_indent_mm": 0.0,
            "text_alignment": "left",
            "chapter_break_mode": "continuous",
        },
        "Minimal": {
            "page_size": "6X9",
            "page_width_mm": 152.4,
            "page_height_mm": 228.6,
            "margin_top_mm": 18.0,
            "margin_bottom_mm": 18.0,
            "margin_left_mm": 18.0,
            "margin_right_mm": 18.0,
            "body_font": "'Noto Sans', sans-serif",
            "heading_font": "'Noto Sans', sans-serif",
            "body_font_size_pt": 11.0,
            "line_height": 1.6,
            "paragraph_spacing_pt": 5.0,
            "first_line_indent_mm": 0.0,
            "text_alignment": "left",
            "chapter_break_mode": "next_page",
        },
        "Large Print": {
            "page_size": "A4",
            "page_width_mm": 210.0,
            "page_height_mm": 297.0,
            "margin_top_mm": 25.0,
            "margin_bottom_mm": 25.0,
            "margin_left_mm": 25.0,
            "margin_right_mm": 25.0,
            "body_font": "'Noto Serif', serif",
            "heading_font": "'Noto Serif', serif",
            "body_font_size_pt": 14.0,
            "line_height": 1.6,
            "paragraph_spacing_pt": 8.0,
            "first_line_indent_mm": 6.0,
            "text_alignment": "left",
            "chapter_break_mode": "next_page",
        }
    }

    @classmethod
    def generate_css(cls, profile: LayoutProfileModel) -> str:
        """
        Generates CSS Paged Media stylesheet for print and preview with safe defaults.
        """
        page_width = profile.page_width_mm or 148.0
        page_height = profile.page_height_mm or 210.0
        margin_top = profile.margin_top_mm or 20.0
        margin_bottom = profile.margin_bottom_mm or 20.0
        margin_left = profile.margin_left_mm or 20.0
        margin_right = profile.margin_right_mm or 20.0
        body_font = profile.body_font or "'Noto Serif'"
        heading_font = profile.heading_font or "'Noto Serif'"
        body_size = profile.body_font_size_pt or 11.0
        line_h = profile.line_height or 1.5
        para_spacing = profile.paragraph_spacing_pt or 4.0
        indent = profile.first_line_indent_mm or 5.0
        alignment = profile.text_alignment or "justify"

        break_rule = "page-break-before: always; break-before: page;"
        if profile.chapter_break_mode == "right_page":
            break_rule = "page-break-before: right; break-before: right;"
        elif profile.chapter_break_mode == "continuous":
            break_rule = "margin-top: 3em;"

        css = f"""
        @page {{
            size: {page_width}mm {page_height}mm;
            margin-top: {margin_top}mm;
            margin-bottom: {margin_bottom}mm;
            margin-left: {margin_left}mm;
            margin-right: {margin_right}mm;
            
            @top-center {{
                content: env(doc-title, string(chapter-title));
                font-family: {heading_font};
                font-size: 8pt;
                color: #666;
            }}
            @bottom-center {{
                content: counter(page);
                font-family: {body_font};
                font-size: 9pt;
                color: #444;
            }}
        }}

        @page:first {{
            @top-center {{ content: normal; }}
            @bottom-center {{ content: normal; }}
        }}

        body {{
            font-family: {body_font}, 'Noto Serif', 'Times New Roman', serif;
            font-size: {body_size}pt;
            line-height: {line_h};
            color: #111;
            text-align: {alignment};
            orphans: 2;
            widows: 2;
            margin: 0;
            padding: 0;
            background-color: #fff;
        }}

        .chapter {{
            {break_rule}
        }}

        .chapter-title {{
            string-set: chapter-title content();
            font-family: {heading_font};
            font-size: {body_size * 1.8}pt;
            font-weight: bold;
            margin-top: 1.5em;
            margin-bottom: 1em;
            text-align: center;
            page-break-after: avoid;
            break-after: avoid;
        }}

        h1, h2, h3, h4 {{
            font-family: {heading_font};
            page-break-after: avoid;
            break-after: avoid;
        }}

        h1 {{ font-size: {body_size * 1.5}pt; margin-top: 1.2em; margin-bottom: 0.6em; }}
        h2 {{ font-size: {body_size * 1.3}pt; margin-top: 1.0em; margin-bottom: 0.5em; }}
        h3 {{ font-size: {body_size * 1.15}pt; margin-top: 0.8em; margin-bottom: 0.4em; }}
        h4 {{ font-size: {body_size * 1.0}pt; font-style: italic; margin-top: 0.6em; margin-bottom: 0.3em; }}

        p {{
            margin-top: 0;
            margin-bottom: {para_spacing}pt;
            text-indent: {indent}mm;
        }}

        .no-indent {{
            text-indent: 0 !important;
        }}

        blockquote {{
            margin: 1em 2em;
            font-style: italic;
            border-left: 3px solid #ccc;
            padding-left: 1em;
        }}

        figure {{
            margin: 1.5em 0;
            text-align: center;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        figure img {{
            max-width: 100%;
            height: auto;
        }}

        figcaption {{
            font-size: {body_size * 0.85}pt;
            color: #555;
            margin-top: 0.5em;
            font-style: italic;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            page-break-inside: avoid;
            break-inside: avoid;
            font-size: {body_size * 0.9}pt;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 6px 10px;
            text-align: left;
        }}

        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}

        .footnote {{
            font-size: {body_size * 0.8}pt;
            color: #444;
            border-top: 1px solid #e0e0e0;
            padding-top: 4px;
            margin-top: 1em;
            text-indent: 0;
        }}
        """
        return css
