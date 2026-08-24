import html
from typing import List, Optional
from app.models.canonical import CanonicalDocument, Chapter, DocumentNode
from app.db.models import LayoutProfileModel
from app.services.renderer.theme_engine import ThemeEngine
from app.services.renderer.semantic_html import SemanticHTMLRenderer


class PreviewService:
    @classmethod
    def render_smart_preview(
        cls,
        doc: CanonicalDocument,
        profile: LayoutProfileModel,
        sample_type: str = "representative",
        target_chapter_id: Optional[str] = None,
        node_limit: int = 40
    ) -> str:
        """
        Renders a lightweight HTML preview representing sample pages
        without generating hundreds of pages in memory.
        """
        selected_nodes: List[DocumentNode] = []
        preview_title = doc.metadata.title or "Xem trước tài liệu"

        if target_chapter_id:
            ch = next((c for c in doc.chapters if c.id == target_chapter_id), None)
            if ch:
                selected_nodes = ch.nodes[:node_limit]
                preview_title = ch.translated_title or ch.title
        elif sample_type == "first_pages" and doc.chapters:
            selected_nodes = doc.chapters[0].nodes[:node_limit]
            preview_title = doc.chapters[0].translated_title or doc.chapters[0].title
        else:
            # Representative sample: Take chapter 1 opening + 1 node with image/table if exists
            if doc.chapters:
                ch0 = doc.chapters[0]
                selected_nodes.extend(ch0.nodes[:15])
                # Find image or table node across other chapters
                for ch in doc.chapters[1:]:
                    for n in ch.nodes:
                        if n.type in ["image", "table"]:
                            selected_nodes.append(n)
                            break
                    if len(selected_nodes) >= node_limit:
                        break

        css = ThemeEngine.generate_css(profile)
        body_html = "".join([SemanticHTMLRenderer.render_node(n) for n in selected_nodes])

        return f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Preview: {html.escape(preview_title)}</title>
    <style>
        {css}
        body {{
            padding: 24px;
            background: #fff;
            max-width: {profile.page_width_mm}mm;
            margin: 0 auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="preview-container">
        <h1 class="chapter-title">{html.escape(preview_title)}</h1>
        {body_html}
    </div>
</body>
</html>"""
