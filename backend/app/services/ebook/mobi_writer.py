"""Ghi MOBI 6 (Mobipocket cổ điển, nén PalmDOC) không cần Calibre.

MOBI 6 là định dạng mọi Kindle đời cũ (K3 trở về sau) và crengine của KOReader đều đọc được. Cấu trúc theo
đặc tả trên MobileRead wiki và cách Calibre ghi bản "old": bản ghi 0 (PalmDOC + MOBI + EXTH), các bản ghi
chữ 4096 byte nén PalmDOC (kèm byte nối ký tự UTF-8 bị cắt ngang), ảnh, FLIS, FCIS, EOF.

Liên kết trong MOBI là vị trí byte (filepos) trong chữ chưa nén: placeholder độ dài cố định được ghi trước rồi
điền vị trí thật sau khi biết toàn bộ văn bản.
"""
from __future__ import annotations

import html
import io
import random
import re
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from app.services.ebook.book_plan import BookPlan, PlanBlock, TocNode
from app.services.ebook.devices import DeviceProfile
from app.services.ebook.images import prepare_image

RECORD_SIZE = 4096
# MOBI 6: Kindle cũ từ chối ảnh lớn hơn 127 KB.
MAX_IMAGE_BYTES = 127 * 1024
MAX_THUMB_BYTES = 16 * 1024
_FILEPOS_WIDTH = 10

# Mã ngôn ngữ (LANGID) mà Kindle dùng trong trường locale.
_MOBI_LOCALES = {
    "vi": 42, "en": 9, "fr": 12, "de": 7, "es": 10, "it": 16, "ja": 17, "ko": 18, "zh": 4,
    "ru": 25, "pt": 22, "nl": 19, "th": 30, "id": 33,
}

_FLIS = b"FLIS\0\0\0\x08\0\x41\0\0\0\0\0\0\xff\xff\xff\xff\0\x01\0\x03\0\0\0\x03\0\0\0\x01" + b"\xff" * 4
_EOF = b"\xe9\x8e\r\n"


# ---------------------------------------------------------------------------------------------
# Nén PalmDOC (LZ77 rút gọn)
# ---------------------------------------------------------------------------------------------

def palmdoc_compress(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        if 10 < i and n - i > 10:
            window_start = max(0, i - 2047)
            if data.rfind(data[i:i + 3], window_start, i) != -1:
                match_pos, match_len = -1, 0
                for length in range(10, 2, -1):
                    pos = data.rfind(data[i:i + length], window_start, i)
                    if pos != -1:
                        match_pos, match_len = pos, length
                        break
                if match_pos != -1:
                    distance = i - match_pos
                    out += struct.pack(">H", 0x8000 | (distance << 3) | (match_len - 3))
                    i += match_len
                    continue
        byte = data[i]
        i += 1
        if byte == 0x20 and i < n and 0x40 <= data[i] < 0x80:
            out.append(data[i] ^ 0x80)
            i += 1
            continue
        if byte == 0 or 0x09 <= byte < 0x80:
            out.append(byte)
            continue
        # Chuỗi byte "lạ" (0x01-0x08, 0x80-0xFF): tối đa 8 byte, tiền tố là độ dài.
        run = bytearray([byte])
        while i < n and len(run) < 8:
            nxt = data[i]
            if nxt == 0 or 0x09 <= nxt < 0x80:
                break
            run.append(nxt)
            i += 1
        out.append(len(run))
        out += run
    return bytes(out)


def palmdoc_decompress(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        c = data[i]
        i += 1
        if 1 <= c <= 8:
            out += data[i:i + c]
            i += c
        elif c < 0x80:
            out.append(c)
        elif c >= 0xC0:
            out.append(0x20)
            out.append(c ^ 0x80)
        else:
            c = (c << 8) | data[i]
            i += 1
            distance = (c >> 3) & 0x07FF
            length = (c & 7) + 3
            for _ in range(length):
                out.append(out[-distance])
    return bytes(out)


def _split_text_records(text: bytes) -> List[Tuple[bytes, bytes]]:
    """Cắt chữ thành bản ghi 4096 byte; phần còn lại của ký tự UTF-8 bị cắt ngang đi kèm làm "overlap"."""
    records = []
    pos = 0
    total = len(text)
    while pos < total:
        end = min(pos + RECORD_SIZE, total)
        chunk = text[pos:end]
        overlap = b""
        if end < total:
            # Lùi tới đầu ký tự cuối cùng của bản ghi rồi lấy các byte nối tiếp thuộc ký tự đó ở bản ghi sau.
            k = end - 1
            while k > pos and (text[k] & 0xC0) == 0x80:
                k -= 1
            lead = text[k]
            need = 1
            if lead >= 0xF0:
                need = 4
            elif lead >= 0xE0:
                need = 3
            elif lead >= 0xC0:
                need = 2
            have = end - k
            if need > have:
                overlap = text[end:end + (need - have)]
        records.append((chunk, overlap))
        pos = end
    return records


# ---------------------------------------------------------------------------------------------
# HTML kiểu Mobipocket
# ---------------------------------------------------------------------------------------------

class _MobiText:
    """Ghép HTML theo byte, ghi nhớ vị trí neo và chỗ cần điền filepos."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.anchors: Dict[str, int] = {}
        self.links: List[Tuple[int, str]] = []

    def write(self, text: str) -> None:
        self.buf += text.encode("utf-8")

    def anchor(self, key: str) -> None:
        self.anchors.setdefault(key, len(self.buf))

    def filepos(self, key: str) -> None:
        self.write("filepos=")
        self.links.append((len(self.buf), key))
        self.write("0" * _FILEPOS_WIDTH)

    def finish(self) -> bytes:
        for offset, key in self.links:
            target = self.anchors.get(key, 0)
            self.buf[offset:offset + _FILEPOS_WIDTH] = f"{target:0{_FILEPOS_WIDTH}d}".encode("ascii")
        return bytes(self.buf)


def _mobi_paragraph_attrs(block: PlanBlock, plan: BookPlan) -> str:
    classes = set(block.classes)
    if "center" in classes or "scene" in classes:
        return ' align="center" width="0"'
    if "right" in classes:
        return ' align="right" width="0"'
    if "verse" in classes or "noindent" in classes or "footnote" in classes:
        return ' width="0"'
    if plan.style.paragraph_style == "block":
        return ' width="0" height="8pt"'
    return ""


def _render_blocks(out: _MobiText, plan: BookPlan, blocks: List[PlanBlock], image_index: Dict[str, int], file_key: str) -> None:
    for block in blocks:
        if block.kind == "heading":
            if block.anchor:
                out.anchor(f"{file_key}#{block.anchor}")
            level = max(1, min(4, block.level))
            align = ""
            if "center" in block.classes or (level == 1 and "left" not in block.classes and "right" not in block.classes):
                align = ' align="center"'
            elif "right" in block.classes:
                align = ' align="right"'
            out.write(f"<h{level}{align}>{block.html}</h{level}>")
        elif block.kind == "para":
            if "footnote" in block.classes:
                out.write(f'<p width="0"><font size="2">{block.html}</font></p>')
            elif "quote" in block.classes:
                out.write(f"<blockquote><p{_mobi_paragraph_attrs(block, plan)}>{block.html}</p></blockquote>")
            else:
                out.write(f"<p{_mobi_paragraph_attrs(block, plan)}>{block.html}</p>")
        elif block.kind == "image":
            index = image_index.get(block.image_key)
            if index is None:
                continue
            out.write(f'<p align="center" width="0"><img recindex="{index:05d}" /></p>')
            if block.html:
                out.write(f'<p align="center" width="0"><i>{block.html}</i></p>')
        elif block.kind == "list":
            tag = "ol" if block.ordered else "ul"
            out.write(f"<{tag}>" + "".join(f"<li>{item}</li>" for item in block.items) + f"</{tag}>")
        elif block.kind == "table":
            rows = []
            for row in block.rows:
                rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
            out.write('<table border="1">' + "".join(rows) + "</table>")
        elif block.kind == "code":
            out.write(f"<pre>{block.html}</pre>")
        elif block.kind == "quote":
            out.write(f"<blockquote>{block.html}</blockquote>")
        elif block.kind == "rule":
            out.write("<hr />")
        elif block.kind == "footnotes":
            out.write("<hr />" + "".join(f'<p width="0"><font size="2">{item}</font></p>' for item in block.items))


def _toc_target(node: TocNode) -> str:
    return f"{node.file_key}#{node.anchor}" if node.anchor else f"{node.file_key}#"


def build_mobi_html(plan: BookPlan, image_index: Dict[str, int]) -> bytes:
    labels = plan.labels
    out = _MobiText()
    out.write("<html><head><guide>")
    out.write(f'<reference type="toc" title="{html.escape(labels["toc"])}" ')
    out.filepos("__toc__")
    out.write(" />")
    out.write(f'<reference type="text" title="{html.escape(labels["start"])}" ')
    out.filepos(f"{plan.files[0].key}#")
    out.write(" /></guide></head><body>")

    flat = plan.flat_toc()
    if len(flat) >= 2:
        out.anchor("__toc__")
        out.write(f'<h2 align="center">{html.escape(labels["toc"])}</h2>')
        for node in flat:
            pad = "&#160;" * (4 * (node.level - 1))
            out.write(f'<p width="0">{pad}<a ')
            out.filepos(_toc_target(node))
            out.write(f">{html.escape(node.title)}</a></p>")
        out.write("<mbp:pagebreak />")
    for index, plan_file in enumerate(plan.files):
        if index:
            out.write("<mbp:pagebreak />")
        out.anchor(f"{plan_file.key}#")
        _render_blocks(out, plan, plan_file.blocks, image_index, plan_file.key)
    if "__toc__" not in out.anchors:
        out.anchors["__toc__"] = out.anchors.get(f"{plan.files[0].key}#", 0)
    out.write("</body></html>")
    return out.finish()


# ---------------------------------------------------------------------------------------------
# Đóng gói PalmDB
# ---------------------------------------------------------------------------------------------

def _exth(records: List[Tuple[int, bytes]]) -> bytes:
    body = b"".join(struct.pack(">II", kind, len(data) + 8) + data for kind, data in records)
    header = b"EXTH" + struct.pack(">II", 12 + len(body), len(records)) + body
    padding = (4 - len(header) % 4) % 4
    return header + b"\0" * padding


def _fcis(text_length: int) -> bytes:
    return (
        b"FCIS\x00\x00\x00\x14\x00\x00\x00\x10\x00\x00\x00\x01\x00\x00\x00\x00"
        + struct.pack(">I", text_length)
        + b"\x00\x00\x00\x00\x00\x00\x00\x20\x00\x00\x00\x08\x00\x01\x00\x01\x00\x00\x00\x00"
    )


def _palm_name(title: str) -> bytes:
    name = re.sub(r"[^-A-Za-z0-9]+", "_", title).strip("_") or "book"
    return name.encode("ascii", "replace")[:31].ljust(32, b"\0")


def _thumbnail(cover_jpeg: bytes) -> Optional[bytes]:
    try:
        im = Image.open(io.BytesIO(cover_jpeg))
        im.thumbnail((180, 240))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=75)
        data = buf.getvalue()
        return data if len(data) <= MAX_THUMB_BYTES * 2 else None
    except Exception:
        return None


def write_mobi(plan: BookPlan, output_path: Path, device: DeviceProfile, grayscale_images: bool = True) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ảnh: JPEG/PNG nhỏ hơn 127 KB; MOBI đánh số ảnh từ 1 theo thứ tự bản ghi.
    image_records: List[bytes] = []
    image_index: Dict[str, int] = {}
    for key, image in plan.images.items():
        data = image.load()
        prepared = prepare_image(data, device.image_max_side, grayscale=grayscale_images,
                                 max_bytes=MAX_IMAGE_BYTES, force_jpeg=True) if data else None
        if prepared and len(prepared.data) <= MAX_IMAGE_BYTES:
            image_records.append(prepared.data)
            image_index[key] = len(image_records)

    cover_offset = thumb_offset = None
    if plan.cover:
        cover = prepare_image(plan.cover, max(device.screen_width, device.screen_height), force_jpeg=True,
                              max_bytes=MAX_IMAGE_BYTES, max_width=device.screen_width,
                              max_height=device.screen_height)
        if cover and len(cover.data) <= MAX_IMAGE_BYTES:
            cover_offset = len(image_records)
            image_records.append(cover.data)
            thumb = _thumbnail(cover.data)
            if thumb:
                thumb_offset = len(image_records)
                image_records.append(thumb)

    text = build_mobi_html(plan, image_index)
    text_records = []
    for chunk, overlap in _split_text_records(text):
        text_records.append(palmdoc_compress(chunk) + overlap + bytes([len(overlap)]))

    n_text = len(text_records)
    first_image = n_text + 1 if image_records else 0xFFFFFFFF
    flis_index = n_text + 1 + len(image_records)
    fcis_index = flis_index + 1
    last_content = n_text + len(image_records)

    title_bytes = plan.title.encode("utf-8")
    exth_records: List[Tuple[int, bytes]] = []
    if plan.author:
        exth_records.append((100, plan.author.encode("utf-8")))
    if plan.translator:
        exth_records.append((108, plan.translator.encode("utf-8")))
    exth_records.append((106, plan.modified[:10].encode("ascii")))
    exth_records.append((501, b"EBOK"))
    exth_records.append((503, title_bytes))
    exth_records.append((524, plan.language.encode("ascii", "ignore") or b"en"))
    if cover_offset is not None:
        exth_records.append((201, struct.pack(">I", cover_offset)))
        exth_records.append((203, struct.pack(">I", 0)))
    if thumb_offset is not None:
        exth_records.append((202, struct.pack(">I", thumb_offset)))
    exth = _exth(exth_records)

    mobi_header_length = 232
    full_name_offset = 16 + mobi_header_length + len(exth)
    locale = _MOBI_LOCALES.get(plan.language.split("-")[0].lower(), 9)

    header = bytearray()
    # PalmDOC header (16 byte): nén PalmDOC, độ dài chữ, số bản ghi chữ, cỡ bản ghi, không mã hóa.
    header += struct.pack(">HHIHHHH", 2, 0, len(text), n_text, RECORD_SIZE, 0, 0)
    header += b"MOBI"
    header += struct.pack(">IIII", mobi_header_length, 2, 65001, random.getrandbits(32))
    header += struct.pack(">I", 6)  # phiên bản file
    header += b"\xff" * 40  # 10 chỉ mục không dùng (orthographic, inflection, index names/keys, extra 0-5)
    header += struct.pack(">I", n_text + 1)  # bản ghi đầu tiên không phải chữ
    header += struct.pack(">II", full_name_offset, len(title_bytes))
    header += struct.pack(">III", locale, 0, 0)  # locale, input/output language
    header += struct.pack(">I", 6)  # phiên bản tối thiểu
    header += struct.pack(">I", first_image)
    header += struct.pack(">IIII", 0, 0, 0, 0)  # Huffman (không dùng)
    header += struct.pack(">I", 0x50)  # cờ EXTH
    header += b"\0" * 32
    header += b"\xff\xff\xff\xff"
    header += struct.pack(">IIII", 0xFFFFFFFF, 0, 0, 0)  # DRM: không có
    header += b"\0" * 8
    header += struct.pack(">HH", 1, last_content)
    header += struct.pack(">IIII", 1, fcis_index, 1, flis_index)
    header += struct.pack(">I", 1)
    header += b"\0" * 8
    header += b"\xff\xff\xff\xff"
    header += struct.pack(">I", 0)
    header += b"\xff\xff\xff\xff"
    header += b"\xff\xff\xff\xff"
    header += struct.pack(">I", 1)  # cờ dữ liệu cuối bản ghi: có byte nối ký tự UTF-8
    header += b"\xff\xff\xff\xff"  # không có chỉ mục INDX
    assert len(header) == 16 + mobi_header_length, len(header)

    record0 = bytes(header) + exth + title_bytes
    record0 += b"\0" * (2 + (4 - (len(record0) + 2) % 4) % 4)

    records = [record0] + text_records + image_records + [_FLIS, _fcis(len(text)), _EOF]

    # PalmDB header (78 byte) + danh sách bản ghi.
    now = int(time.time())
    palm = bytearray()
    palm += _palm_name(plan.title)
    palm += struct.pack(">HHIIIIII", 0, 0, now, now, 0, 0, 0, 0)
    palm += b"BOOKMOBI"
    palm += struct.pack(">II", 2 * len(records) - 1, 0)
    palm += struct.pack(">H", len(records))
    offset = len(palm) + 8 * len(records) + 2
    for index, record in enumerate(records):
        palm += struct.pack(">I", offset)
        palm += struct.pack(">I", 2 * index)[1:].rjust(4, b"\0")
        offset += len(record)
    palm += b"\0\0"

    tmp_path = output_path.with_name(output_path.name + ".part")
    with open(tmp_path, "wb") as fh:
        fh.write(palm)
        for record in records:
            fh.write(record)
    tmp_path.replace(output_path)
    return output_path
