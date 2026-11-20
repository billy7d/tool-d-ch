"""Ghi EPUB 3 kèm phần tương thích EPUB 2 (toc.ncx, guide), tối ưu cho KOReader (crengine) trên Kindle cũ.

- CSS tối giản, chỉ dùng em: không ép font, cỡ chữ hay giãn dòng, để KOReader áp cài đặt của người đọc;
  không ép font nên chữ Việt luôn hiện bằng font có đủ dấu của KOReader.
- Mỗi chương một file XHTML (KOReader mở trang mới ở mỗi file), file quá lớn được chia nhỏ.
- lang/xml:lang đúng ngôn ngữ sách để KOReader chọn đúng bộ ngắt dòng (tiếng Việt không ngắt âm tiết).
- Ảnh JPEG baseline/PNG không trong suốt, không CMYK, vừa độ phân giải máy.
- Bìa khai báo theo cả hai kiểu (meta cover + cover-image) để hiện trong trình duyệt sách của KOReader.
"""
from __future__ import annotations

import html
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from app.services.ebook.book_plan import BookPlan, PlanBlock, TocNode
from app.services.ebook.devices import DeviceProfile
from app.services.ebook.images import PreparedImage, prepare_image


def _esc(text: str) -> str:
    return html.escape(text or "", quote=True)


def build_css(plan: BookPlan) -> str:
    style = plan.style
    body_align = "justify" if style.justify else "left"
    if style.paragraph_style == "indent":
        paragraph = f"p {{ margin: 0; text-indent: {style.indent_em:g}em; }}"
    else:
        paragraph = "p { margin: 0 0 0.6em 0; text-indent: 0; }"
    return f"""/* Kiểu chữ tối giản cho KOReader / máy đọc e-ink: không ép font, cỡ chữ hay giãn dòng. */
html, body {{ margin: 0; padding: 0; }}
body {{ text-align: {body_align}; }}
{paragraph}
p.noindent, p.center, p.right, p.verse, p.scene, p.footnote, p.caption, p.toc-entry {{ text-indent: 0; }}
p.center {{ text-align: center; }}
p.right {{ text-align: right; }}
p.verse {{ text-align: left; margin: 0.6em 0 0.6em 1.5em; }}
p.verse.center {{ text-align: center; margin-left: 0; }}
p.scene {{ text-align: center; margin: 0.8em 0; }}
p.quote {{ margin-left: 1.5em; margin-right: 1.5em; }}
h1, h2, h3, h4 {{ text-indent: 0; font-weight: bold; line-height: 1.25; page-break-after: avoid; break-after: avoid; hyphens: none; -webkit-hyphens: none; adobe-hyphenate: none; }}
h1 {{ font-size: 1.5em; margin: 1.5em 0 1em 0; text-align: center; }}
h2 {{ font-size: 1.3em; margin: 1.3em 0 0.6em 0; text-align: left; }}
h3 {{ font-size: 1.15em; margin: 1.1em 0 0.5em 0; text-align: left; }}
h4 {{ font-size: 1em; margin: 1em 0 0.4em 0; text-align: left; }}
h1.left, h2.left, h3.left, h4.left {{ text-align: left; }}
h1.center, h2.center, h3.center, h4.center {{ text-align: center; }}
h1.right, h2.right, h3.right, h4.right {{ text-align: right; }}
div.figure {{ text-align: center; text-indent: 0; margin: 0.8em 0; page-break-inside: avoid; break-inside: avoid; }}
div.figure img {{ max-width: 100%; height: auto; }}
p.caption {{ text-align: center; font-size: 0.9em; font-style: italic; margin: 0.3em 0 0 0; }}
blockquote {{ margin: 0.6em 1.5em; }}
blockquote p {{ text-indent: 0; }}
ul, ol {{ margin: 0.5em 0 0.5em 1.5em; padding: 0; }}
li {{ margin: 0.15em 0; }}
pre {{ white-space: pre-wrap; font-family: monospace; font-size: 0.85em; text-align: left; margin: 0.6em 0; }}
table {{ border-collapse: collapse; margin: 0.8em 0; font-size: 0.9em; }}
th, td {{ border: 1px solid #555; padding: 0.2em 0.4em; text-align: left; vertical-align: top; }}
hr {{ border: 0; border-top: 1px solid #777; margin: 1em 25%; }}
div.footnotes {{ margin-top: 1.5em; border-top: 1px solid #777; padding-top: 0.4em; }}
p.footnote {{ font-size: 0.85em; margin: 0 0 0.4em 0; text-align: left; }}
sup {{ font-size: 0.7em; vertical-align: super; line-height: 0; }}
div.cover {{ text-align: center; margin: 0; padding: 0; }}
div.cover img {{ max-width: 100%; max-height: 100%; }}
nav ol {{ list-style-type: none; margin: 0 0 0 1em; padding: 0; }}
nav > ol {{ margin-left: 0; }}
nav li {{ margin: 0.3em 0; }}
h1.toc-title {{ text-align: center; }}
.hidden {{ display: none; }}
"""


def _xhtml(title: str, body: str, language: str, css_href: str = "../styles/book.css") -> str:
    lang = _esc(language)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!DOCTYPE html>\n"
        f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{lang}" lang="{lang}">\n'
        "<head>\n"
        f"<title>{_esc(title)}</title>\n"
        f'<link rel="stylesheet" type="text/css" href="{css_href}"/>\n'
        "</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def _class_attr(classes) -> str:
    return f' class="{" ".join(classes)}"' if classes else ""


def _render_block(block: PlanBlock, images: Dict[str, PreparedImage]) -> str:
    if block.kind == "heading":
        level = max(1, min(4, block.level))
        anchor = f' id="{block.anchor}"' if block.anchor else ""
        return f"<h{level}{anchor}{_class_attr(block.classes)}>{block.html}</h{level}>"
    if block.kind == "para":
        return f"<p{_class_attr(block.classes)}>{block.html}</p>"
    if block.kind == "image":
        image = images.get(block.image_key)
        if not image:
            return ""
        caption = f'\n<p class="caption">{block.html}</p>' if block.html else ""
        return (f'<div class="figure"><img src="../images/{block.image_key}.{image.ext}" '
                f'alt="{_esc(block.text)}"/>{caption}</div>')
    if block.kind == "list":
        tag = "ol" if block.ordered else "ul"
        items = "".join(f"<li>{item}</li>" for item in block.items)
        return f"<{tag}>{items}</{tag}>"
    if block.kind == "table":
        rows = []
        for index, row in enumerate(block.rows):
            cell = "th" if index == 0 and len(block.rows) > 1 else "td"
            rows.append("<tr>" + "".join(f"<{cell}>{c}</{cell}>" for c in row) + "</tr>")
        return "<table>" + "".join(rows) + "</table>"
    if block.kind == "code":
        return f"<pre>{block.html}</pre>"
    if block.kind == "quote":
        return f"<blockquote><p>{block.html}</p></blockquote>"
    if block.kind == "rule":
        return "<hr/>"
    if block.kind == "footnotes":
        notes = "".join(f'<p class="footnote">{item}</p>' for item in block.items)
        return f'<div class="footnotes">{notes}</div>'
    return ""


def _href(node: TocNode) -> str:
    return f"text/{node.file_key}.xhtml" + (f"#{node.anchor}" if node.anchor else "")


def _nav_list(nodes: List[TocNode], prefix: str = "") -> str:
    items = []
    for node in nodes:
        children = _nav_list(node.children, prefix) if node.children else ""
        items.append(f'<li><a href="{prefix}{_href(node)}">{_esc(node.title)}</a>{children}</li>')
    return "<ol>" + "".join(items) + "</ol>"


def _ncx(plan: BookPlan) -> str:
    counter = [0]

    def points(nodes: List[TocNode]) -> str:
        out = []
        for node in nodes:
            counter[0] += 1
            order = counter[0]
            out.append(
                f'<navPoint id="np{order}" playOrder="{order}"><navLabel><text>{_esc(node.title)}</text></navLabel>'
                f'<content src="{_href(node)}"/>{points(node.children)}</navPoint>'
            )
        return "".join(out)

    nav_points = points(plan.toc)
    author = f"<docAuthor><text>{_esc(plan.author)}</text></docAuthor>" if plan.author else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1" xml:lang="{_esc(plan.language)}">\n'
        "<head>"
        f'<meta name="dtb:uid" content="{_esc(plan.identifier)}"/>'
        f'<meta name="dtb:depth" content="{plan.toc_depth()}"/>'
        '<meta name="dtb:totalPageCount" content="0"/><meta name="dtb:maxPageNumber" content="0"/>'
        "</head>\n"
        f"<docTitle><text>{_esc(plan.title)}</text></docTitle>{author}\n"
        f"<navMap>{nav_points}</navMap>\n</ncx>\n"
    )


def _opf(plan: BookPlan, manifest: List[str], spine: List[str], guide: List[str], has_cover: bool) -> str:
    meta = [
        f'<dc:identifier id="bookid">{_esc(plan.identifier)}</dc:identifier>',
        f"<dc:title>{_esc(plan.title)}</dc:title>",
        f"<dc:language>{_esc(plan.language)}</dc:language>",
        f'<meta property="dcterms:modified">{plan.modified}</meta>',
    ]
    if plan.author:
        meta.append(f'<dc:creator id="creator">{_esc(plan.author)}</dc:creator>')
    if plan.translator:
        meta.append(f'<dc:contributor id="translator">{_esc(plan.translator)}</dc:contributor>')
        meta.append('<meta refines="#translator" property="role" scheme="marc:relators">trl</meta>')
    if has_cover:
        meta.append('<meta name="cover" content="cover-image"/>')
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="{_esc(plan.language)}">\n'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">\n'
        + "\n".join(meta) + "\n</metadata>\n"
        "<manifest>\n" + "\n".join(manifest) + "\n</manifest>\n"
        '<spine toc="ncx">\n' + "\n".join(spine) + "\n</spine>\n"
        "<guide>\n" + "\n".join(guide) + "\n</guide>\n"
        "</package>\n"
    )


def write_epub(plan: BookPlan, output_path: Path, device: DeviceProfile, grayscale_images: bool = False) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = plan.labels

    prepared: Dict[str, PreparedImage] = {}
    for key, image in plan.images.items():
        data = image.load()
        if not data:
            continue
        result = prepare_image(data, device.image_max_side, grayscale=grayscale_images)
        if result:
            prepared[key] = result

    cover: Optional[PreparedImage] = None
    if plan.cover:
        cover = prepare_image(plan.cover, max(device.screen_width, device.screen_height), force_jpeg=True,
                              max_width=device.screen_width, max_height=device.screen_height)

    manifest = [
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="css" href="styles/book.css" media-type="text/css"/>',
    ]
    spine: List[str] = []
    guide: List[str] = []
    first_key = plan.files[0].key

    tmp_path = output_path.with_name(output_path.name + ".part")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # mimetype phải là mục đầu tiên, không nén.
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>'
            "</container>\n",
        )
        zf.writestr("OEBPS/styles/book.css", build_css(plan))

        if cover:
            zf.writestr(f"OEBPS/images/cover.{cover.ext}", cover.data)
            manifest.append(f'<item id="cover-image" href="images/cover.{cover.ext}" media-type="{cover.media_type}" properties="cover-image"/>')
            cover_body = f'<div class="cover"><img src="../images/cover.{cover.ext}" alt="{_esc(labels["cover"])}"/></div>'
            zf.writestr("OEBPS/text/cover.xhtml", _xhtml(labels["cover"], cover_body, plan.language))
            manifest.append('<item id="cover" href="text/cover.xhtml" media-type="application/xhtml+xml"/>')
            spine.append('<itemref idref="cover"/>')
            guide.append(f'<reference type="cover" title="{_esc(labels["cover"])}" href="text/cover.xhtml"/>')

        show_toc_page = len(plan.flat_toc()) >= 2
        if show_toc_page:
            spine.append('<itemref idref="nav"/>')
            guide.append(f'<reference type="toc" title="{_esc(labels["toc"])}" href="nav.xhtml"/>')
        landmarks = [f'<li><a epub:type="bodymatter" href="text/{first_key}.xhtml">{_esc(labels["start"])}</a></li>']
        if cover:
            landmarks.insert(0, f'<li><a epub:type="cover" href="text/cover.xhtml">{_esc(labels["cover"])}</a></li>')
        nav_body = (
            f'<nav epub:type="toc" id="toc"><h1 class="toc-title">{_esc(labels["toc"])}</h1>{_nav_list(plan.toc)}</nav>\n'
            f'<nav epub:type="landmarks" id="landmarks" class="hidden" hidden="hidden"><ol>{"".join(landmarks)}</ol></nav>'
        )
        zf.writestr("OEBPS/nav.xhtml", _xhtml(labels["toc"], nav_body, plan.language, css_href="styles/book.css"))

        for key, image in prepared.items():
            zf.writestr(f"OEBPS/images/{key}.{image.ext}", image.data)
            manifest.append(f'<item id="{key}" href="images/{key}.{image.ext}" media-type="{image.media_type}"/>')

        titles = {}
        for node in plan.flat_toc():
            titles.setdefault(node.file_key, node.title)
        current_title = plan.title
        for plan_file in plan.files:
            current_title = titles.get(plan_file.key, current_title)
            body = "\n".join(filter(None, (_render_block(b, prepared) for b in plan_file.blocks)))
            zf.writestr(f"OEBPS/text/{plan_file.key}.xhtml", _xhtml(current_title, body, plan.language))
            manifest.append(f'<item id="{plan_file.key}" href="text/{plan_file.key}.xhtml" media-type="application/xhtml+xml"/>')
            spine.append(f'<itemref idref="{plan_file.key}"/>')
        guide.append(f'<reference type="text" title="{_esc(labels["start"])}" href="text/{first_key}.xhtml"/>')

        zf.writestr("OEBPS/toc.ncx", _ncx(plan))
        zf.writestr("OEBPS/content.opf", _opf(plan, manifest, spine, guide, has_cover=bool(cover)))

    tmp_path.replace(output_path)
    return output_path
