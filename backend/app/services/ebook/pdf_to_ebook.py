"""Chuyển thẳng một file PDF sang EPUB hoặc MOBI (không dịch), tối ưu cho KOReader trên Kindle đời cũ."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from app.models.canonical import CanonicalDocument, NodeType
from app.services.ebook.book_plan import BookStyle, build_book_plan
from app.services.ebook.devices import DeviceProfile, get_device
from app.services.ebook.epub_writer import write_epub
from app.services.ebook.mobi_writer import write_mobi
from app.services.ebook.pdf_layout import PdfAnalysis, PdfLayoutAnalyzer
from app.services.reconstruction.structure_builder import StructureBuilder

ProgressFn = Callable[[float, str], None]
SUPPORTED_FORMATS = ("epub", "mobi")
MOBI_ENGINES = ("auto", "builtin", "calibre")


@dataclass
class ConversionOptions:
    output_format: str = "epub"
    device: str = "kindle_pw"
    title: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    include_images: bool = True
    cover: bool = True
    grayscale_images: Optional[bool] = None  # None: EPUB giữ màu, MOBI chuyển xám
    mobi_engine: str = "auto"
    source_name: Optional[str] = None  # tên file người dùng tải lên, làm tên sách khi PDF không ghi tiêu đề


@dataclass
class ConversionResult:
    output_path: str
    output_format: str
    engine: str
    title: str
    author: str
    language: str
    device: str
    page_count: int
    chapter_count: int
    toc_entries: int
    paragraph_count: int
    image_count: int
    scanned_pages: int
    ocr_pages: int
    paragraph_style: str
    outline_used: bool
    file_size_bytes: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def calibre_executable() -> Optional[str]:
    return shutil.which("ebook-convert")


def convert_epub_to_mobi_with_calibre(epub_path: Path, mobi_path: Path, device: DeviceProfile, language: str) -> None:
    """MOBI 6 bằng Calibre: --mobi-file-type old cho Kindle cũ/KOReader, bỏ mục lục chèn thêm (EPUB đã có)."""
    exe = calibre_executable()
    if not exe:
        raise RuntimeError("Máy chưa cài Calibre (lệnh 'ebook-convert').")
    cmd = [
        exe, str(epub_path), str(mobi_path),
        "--output-profile", device.calibre_profile,
        "--mobi-file-type", "old",
        "--no-inline-toc",
        "--language", language,
    ]
    completed = subprocess.run(cmd, capture_output=True, timeout=900)
    if completed.returncode != 0 or not mobi_path.exists():
        detail = (completed.stderr or completed.stdout or b"").decode("utf-8", "replace")[-500:]
        raise RuntimeError(f"Calibre chuyển MOBI thất bại: {detail}")


def _count_paragraphs(doc: CanonicalDocument) -> int:
    return sum(1 for ch in doc.chapters for n in ch.nodes if n.type == NodeType.PARAGRAPH)


def build_canonical_from_pdf(pdf_path: Path, work_dir: Path, options: ConversionOptions,
                             progress: Optional[ProgressFn] = None) -> tuple[CanonicalDocument, PdfAnalysis]:
    device = get_device(options.device)
    analyzer = PdfLayoutAnalyzer(
        work_dir=work_dir,
        device=device,
        include_images=options.include_images,
        cover=options.cover,
        language=options.language,
        title=options.title,
        author=options.author,
        progress=progress,
        source_name=options.source_name,
    )
    analysis = analyzer.analyze(Path(pdf_path))
    doc = StructureBuilder.build_canonical_document(
        project_id=f"pdf-{Path(pdf_path).stem}",
        filename=Path(pdf_path).name,
        pages_data=analysis.pages,
        assets_data=analysis.assets,
        detect_chapters_by_style=not analysis.outline_used,
        preserve_block_order=True,
        filter_headers_footers=False,
        strict_chapter_patterns=True,
        join_page_breaks=False,  # bộ phân tích bố cục đã nối đoạn sang trang (có xét thụt lề, thơ, trích dẫn)
    )
    doc.metadata.title = analysis.title
    doc.metadata.author = analysis.author or None
    doc.metadata.language = analysis.language
    doc.metadata.translator = None
    return doc, analysis


def convert_pdf_to_ebook(
    pdf_path: Path,
    output_path: Path,
    work_dir: Path,
    options: Optional[ConversionOptions] = None,
    progress: Optional[ProgressFn] = None,
) -> ConversionResult:
    options = options or ConversionOptions()
    fmt = (options.output_format or "epub").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Định dạng '{fmt}' không được hỗ trợ (chỉ EPUB hoặc MOBI).")
    engine_choice = (options.mobi_engine or "auto").lower()
    if engine_choice not in MOBI_ENGINES:
        engine_choice = "auto"
    report = progress or (lambda fraction, message: None)
    device = get_device(options.device)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    report(0.02, "Mở file PDF")
    doc, analysis = build_canonical_from_pdf(pdf_path, work_dir, options, progress=report)
    report(0.85, "Dựng chương và mục lục")

    plan = build_book_plan(
        doc,
        language=analysis.language,
        style=BookStyle(paragraph_style=analysis.paragraph_style, indent_em=analysis.indent_em,
                        justify=analysis.justify),
        cover=analysis.cover,
        title=analysis.title,
        author=analysis.author,
    )

    warnings = list(analysis.warnings)
    engine = "builtin"
    if fmt == "epub":
        report(0.9, "Đóng gói EPUB")
        write_epub(plan, output_path, device, grayscale_images=bool(options.grayscale_images))
    else:
        grayscale = True if options.grayscale_images is None else options.grayscale_images
        use_calibre = engine_choice == "calibre" or (engine_choice == "auto" and calibre_executable())
        if use_calibre:
            report(0.9, "Chuyển MOBI bằng Calibre")
            temp_epub = work_dir / "calibre_input.epub"
            write_epub(plan, temp_epub, device, grayscale_images=grayscale)
            try:
                convert_epub_to_mobi_with_calibre(temp_epub, output_path, device, plan.language)
                engine = "calibre"
            except Exception as exc:
                if engine_choice == "calibre":
                    raise
                warnings.append(f"Calibre lỗi, đã dùng bộ ghi MOBI tích hợp: {exc}")
                write_mobi(plan, output_path, device, grayscale_images=grayscale)
            finally:
                temp_epub.unlink(missing_ok=True)
        else:
            report(0.9, "Đóng gói MOBI")
            write_mobi(plan, output_path, device, grayscale_images=grayscale)

    report(0.99, "Kiểm tra file")
    from app.services.exporter.validator import ExportValidator

    valid, message = ExportValidator.validate_export_file(output_path, fmt)
    if not valid:
        raise RuntimeError(f"File {fmt.upper()} tạo ra không hợp lệ: {message}")

    return ConversionResult(
        output_path=str(output_path),
        output_format=fmt,
        engine=engine,
        title=plan.title,
        author=plan.author,
        language=plan.language,
        device=device.key,
        page_count=analysis.page_count,
        chapter_count=len(plan.files),
        toc_entries=len(plan.flat_toc()),
        paragraph_count=_count_paragraphs(doc),
        image_count=len(plan.images),
        scanned_pages=analysis.scanned_pages,
        ocr_pages=analysis.ocr_pages,
        paragraph_style=analysis.paragraph_style,
        outline_used=analysis.outline_used,
        file_size_bytes=output_path.stat().st_size,
        warnings=warnings,
    )
