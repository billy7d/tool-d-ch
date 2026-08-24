import html
from typing import List, Optional
from app.models.canonical import CanonicalDocument, Chapter, DocumentNode, NodeType
from app.db.models import LayoutProfileModel
from app.services.renderer.theme_engine import ThemeEngine


class SemanticHTMLRenderer:
    @classmethod
    def render_node(cls, node: DocumentNode) -> str:
        # Prefer translated content, fallback to original if untranslated
        content = node.translated_content or node.content
        escaped_content = html.escape(content).replace("\n", "<br>")

        if node.type == NodeType.HEADING:
            lvl = node.metadata.heading_level or 1
            return f"<h{lvl} id=\"{node.id}\">{escaped_content}</h{lvl}>"

        elif node.type == NodeType.PARAGRAPH:
            return f"<p id=\"{node.id}\">{escaped_content}</p>"

        elif node.type == NodeType.QUOTE:
            return f"<blockquote id=\"{node.id}\"><p>{escaped_content}</p></blockquote>"

        elif node.type == NodeType.IMAGE:
            asset_path = node.metadata.image_asset_id or ""
            caption = html.escape(node.metadata.image_caption or "")
            cap_html = f"<figcaption>{caption}</figcaption>" if caption else ""
            return f"<figure id=\"{node.id}\"><img src=\"/api/assets/{asset_path}\" alt=\"Illustration\">{cap_html}</figure>"

        elif node.type == NodeType.TABLE:
            rows = node.metadata.table_rows or []
            if not rows:
                return f"<p id=\"{node.id}\">{escaped_content}</p>"
            
            table_lines = [f"<table id=\"{node.id}\">"]
            for r_idx, row in enumerate(rows):
                table_lines.append("<tr>")
                tag = "th" if r_idx == 0 else "td"
                for cell in row:
                    table_lines.append(f"<{tag}>{html.escape(str(cell))}</{tag}>")
                table_lines.append("</tr>")
            table_lines.append("</table>")
            return "".join(table_lines)

        elif node.type == NodeType.FOOTNOTE:
            num = node.metadata.footnote_number or "*"
            return f"<div class=\"footnote\" id=\"{node.id}\"><sup>{num}</sup> {escaped_content}</div>"

        elif node.type == NodeType.CODE_BLOCK:
            lang = node.metadata.code_language or "text"
            return f"<pre id=\"{node.id}\"><code class=\"language-{lang}\">{escaped_content}</code></pre>"

        elif node.type == NodeType.HORIZONTAL_RULE:
            return "<hr>"

        elif node.type == NodeType.PAGE_BREAK_HINT:
            return "<div class=\"page-break\"></div>"

        return f"<p id=\"{node.id}\">{escaped_content}</p>"

    @classmethod
    def render_chapter(cls, chapter: Chapter) -> str:
        title = chapter.translated_title or chapter.title
        num_str = f"Chương {chapter.number}: " if chapter.number else ""
        header_html = f"<h1 class=\"chapter-title\">{html.escape(num_str + title)}</h1>"

        nodes_html = [cls.render_node(n) for n in chapter.nodes]

        return f"""
        <article class="chapter" id="{chapter.id}">
            {header_html}
            {"".join(nodes_html)}
        </article>
        """

    @classmethod
    def render_full_document(cls, doc: CanonicalDocument, profile: LayoutProfileModel) -> str:
        css = ThemeEngine.generate_css(profile)
        chapters_html = [cls.render_chapter(ch) for ch in doc.chapters]

        title = doc.metadata.title or "Document"
        return f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>{html.escape(title)}</title>
    <style>
        {css}
    </style>
</head>
<body>
    {"".join(chapters_html)}
</body>
</html>"""
