import os
import subprocess
from pathlib import Path
from typing import Optional
from app.models.canonical import CanonicalDocument, Chapter, NodeType
from app.db.models import LayoutProfileModel
from app.services.renderer.semantic_html import SemanticHTMLRenderer
from app.services.exporter.validator import ExportValidator


class PDFExporter:
    @classmethod
    def export_pdf(
        cls,
        doc: CanonicalDocument,
        profile: LayoutProfileModel,
        output_pdf_path: Path,
        metadata: Optional[dict] = None
    ) -> Path:
        """
        Renders Canonical Document to a high quality Reflowed Vietnamese PDF.
        """
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        html_content = SemanticHTMLRenderer.render_full_document(doc, profile)
        temp_html_path = output_pdf_path.parent / f"{output_pdf_path.stem}_render.html"

        with open(temp_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # 1. Try WeasyPrint first
        weasyprint_success = False
        try:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(str(output_pdf_path))
            weasyprint_success = True
        except Exception as e:
            print(f"[PDFExporter] WeasyPrint not available or failed ({e}), trying fallback engines...")

        # 2. Try Headless Chrome/Edge if WeasyPrint is unavailable on Windows
        if not weasyprint_success:
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            ]
            
            for browser_exe in chrome_paths:
                if Path(browser_exe).exists():
                    try:
                        cmd = [
                            browser_exe,
                            "--headless=new",
                            "--disable-gpu",
                            f"--print-to-pdf={output_pdf_path.resolve().as_posix()}",
                            temp_html_path.resolve().as_uri()
                        ]
                        subprocess.run(cmd, check=True, timeout=60)
                        weasyprint_success = True
                        break
                    except Exception as ex:
                        print(f"[PDFExporter] Browser print failed: {ex}")

        # 3. Fallback to ReportLab if needed
        if not weasyprint_success or not output_pdf_path.exists():
            cls._render_reportlab_fallback(doc, profile, output_pdf_path)

        # Cleanup temp html
        if temp_html_path.exists():
            temp_html_path.unlink()

        # Validate result
        valid, msg = ExportValidator.validate_export_file(output_pdf_path, "pdf")
        if not valid:
            raise RuntimeError(f"Lỗi xuất PDF: {msg}")

        return output_pdf_path

    @classmethod
    def _render_reportlab_fallback(cls, doc: CanonicalDocument, profile: LayoutProfileModel, output_path: Path):
        """Pure Python fallback PDF generator using ReportLab with UTF-8 support."""
        try:
            from reportlab.lib.pagesizes import A4, A5, letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
            # Default của SQLAlchemy chỉ được gán khi flush; fallback phải tự bảo vệ object tạm.
            pagesize = A5 if (profile.page_size or "A5") == "A5" else A4
            margin_right = profile.margin_right_mm if profile.margin_right_mm is not None else 20.0
            margin_left = profile.margin_left_mm if profile.margin_left_mm is not None else 20.0
            margin_top = profile.margin_top_mm if profile.margin_top_mm is not None else 20.0
            margin_bottom = profile.margin_bottom_mm if profile.margin_bottom_mm is not None else 20.0
            body_font_size = profile.body_font_size_pt if profile.body_font_size_pt is not None else 11.0
            line_height = profile.line_height if profile.line_height is not None else 1.5
            first_line_indent = profile.first_line_indent_mm if profile.first_line_indent_mm is not None else 5.0
            paragraph_spacing = profile.paragraph_spacing_pt if profile.paragraph_spacing_pt is not None else 4.0
            doc_template = SimpleDocTemplate(
                str(output_path),
                pagesize=pagesize,
                rightMargin=margin_right * 2.83,
                leftMargin=margin_left * 2.83,
                topMargin=margin_top * 2.83,
                bottomMargin=margin_bottom * 2.83
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'ChapterTitle',
                parent=styles['Heading1'],
                fontSize=body_font_size * 1.6,
                leading=body_font_size * 2.0,
                alignment=1,  # Căn giữa
                spaceAfter=15
            )
            body_style = ParagraphStyle(
                'Body',
                parent=styles['Normal'],
                fontSize=body_font_size,
                leading=body_font_size * line_height,
                firstLineIndent=first_line_indent * 2.83,
                spaceAfter=paragraph_spacing
            )

            story = []
            for ch in doc.chapters:
                ch_title = ch.translated_title or ch.title
                num_str = f"Chương {ch.number}: " if ch.number else ""
                story.append(Paragraph(f"<b>{num_str}{ch_title}</b>", title_style))
                story.append(Spacer(1, 10))

                for n in ch.nodes:
                    text = (n.translated_content or n.content).replace("\n", "<br/>")
                    if n.type == NodeType.HEADING:
                        lvl_style = ParagraphStyle(
                            'Heading',
                            parent=title_style,
                            fontSize=body_font_size * 1.3,
                            alignment=0,
                            spaceBefore=10,
                            spaceAfter=5
                        )
                        story.append(Paragraph(f"<b>{text}</b>", lvl_style))
                    elif n.type == NodeType.PARAGRAPH:
                        story.append(Paragraph(text, body_style))

                story.append(PageBreak())

            doc_template.build(story)
        except Exception as e:
            raise RuntimeError(f"Không thể khởi tạo engine PDF ReportLab: {e}")
