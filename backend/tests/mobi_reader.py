"""Bộ đọc MOBI 6 tối giản dùng cho test: tách bản ghi PalmDB, đọc header MOBI/EXTH, giải nén chữ PalmDOC.

Viết độc lập với bộ ghi (không import mobi_writer) để test bắt được lỗi định dạng thay vì tự xác nhận chính nó.
"""
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List


def _decompress(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data):
        c = data[i]
        i += 1
        if 1 <= c <= 8:
            out += data[i:i + c]
            i += c
        elif c < 0x80:
            out.append(c)
        elif c >= 0xC0:
            out += b" " + bytes([c ^ 0x80])
        else:
            pair = (c << 8) | data[i]
            i += 1
            distance, length = (pair >> 3) & 0x7FF, (pair & 7) + 3
            for _ in range(length):
                out.append(out[-distance])
    return bytes(out)


def _trailing_size(record: bytes, flags: int) -> int:
    size = 0
    for bit in range(1, 16):
        if flags & (1 << bit):
            value, shift, pos = 0, 0, len(record) - size
            while True:
                byte = record[pos - 1]
                value |= (byte & 0x7F) << shift
                shift += 7
                pos -= 1
                if byte & 0x80 or shift >= 28 or pos == 0:
                    break
            size += value
    if flags & 1:
        size += (record[len(record) - size - 1] & 0x3) + 1
    return size


@dataclass
class MobiBook:
    name: bytes
    records: List[bytes]
    compression: int
    text_length: int
    text_record_count: int
    mobi_header_length: int
    encoding: int
    file_version: int
    first_image_index: int
    extra_flags: int
    full_name: str
    locale: int
    exth: Dict[int, List[bytes]] = field(default_factory=dict)
    text: bytes = b""


def read_mobi(raw: bytes) -> MobiBook:
    assert raw[60:68] == b"BOOKMOBI", "thiếu nhãn BOOKMOBI"
    count = struct.unpack(">H", raw[76:78])[0]
    offsets = [struct.unpack(">I", raw[78 + 8 * i:82 + 8 * i])[0] for i in range(count)]
    offsets.append(len(raw))
    records = [raw[offsets[i]:offsets[i + 1]] for i in range(count)]
    r0 = records[0]
    compression, _, text_length, text_count, record_size = struct.unpack(">HHIHH", r0[:12])
    assert record_size == 4096
    assert r0[16:20] == b"MOBI"
    header_length, _mobi_type, encoding, _uid, version = struct.unpack(">IIIII", r0[20:40])
    full_name_offset, full_name_length, locale = struct.unpack(">III", r0[84:96])
    first_image = struct.unpack(">I", r0[108:112])[0]
    exth_flags = struct.unpack(">I", r0[128:132])[0]
    extra_flags = struct.unpack(">H", r0[0xF2:0xF4])[0]
    exth: Dict[int, List[bytes]] = {}
    if exth_flags & 0x40:
        start = 16 + header_length
        assert r0[start:start + 4] == b"EXTH"
        _, n = struct.unpack(">II", r0[start + 4:start + 12])
        pos = start + 12
        for _ in range(n):
            kind, length = struct.unpack(">II", r0[pos:pos + 8])
            exth.setdefault(kind, []).append(r0[pos + 8:pos + length])
            pos += length
    text = bytearray()
    for record in records[1:1 + text_count]:
        body = record[:len(record) - _trailing_size(record, extra_flags)]
        text += _decompress(body) if compression == 2 else body
    return MobiBook(
        name=raw[:32].rstrip(b"\0"), records=records, compression=compression, text_length=text_length,
        text_record_count=text_count, mobi_header_length=header_length, encoding=encoding,
        file_version=version, first_image_index=first_image, extra_flags=extra_flags,
        full_name=r0[full_name_offset:full_name_offset + full_name_length].decode("utf-8"),
        locale=locale, exth=exth, text=bytes(text),
    )


def filepos_targets(text: bytes) -> List[int]:
    return [int(m.group(1)) for m in re.finditer(rb"filepos=(\d{10})", text)]
