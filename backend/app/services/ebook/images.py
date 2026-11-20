"""Chuẩn hóa ảnh cho máy đọc e-ink đời cũ.

Kindle cũ không vẽ được ảnh CMYK, JPEG progressive hay JPEG2000, ảnh PNG trong suốt hiện nền đen,
và ảnh vài nghìn pixel làm KOReader trên máy RAM 256 MB giải mã rất chậm.
"""
import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000


@dataclass
class PreparedImage:
    data: bytes
    media_type: str
    ext: str
    width: int
    height: int


def _flatten(im: Image.Image) -> Image.Image:
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    if im.mode in ("1", "I;16", "I;16B", "I;16L", "I", "F"):
        return im.convert("L")
    if im.mode not in ("L", "RGB"):
        return im.convert("RGB")
    return im


def _is_line_art(im: Image.Image) -> bool:
    """Ảnh ít màu (sơ đồ, chữ, nét vẽ) giữ PNG cho nét sắc; ảnh chụp dùng JPEG cho nhẹ."""
    limit = 16 if im.mode == "L" else 64
    try:
        return im.getcolors(maxcolors=limit) is not None
    except Exception:
        return False


def _encode_jpeg(im: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=False)
    return buf.getvalue()


def prepare_image(
    data: bytes,
    max_side: int,
    grayscale: bool = False,
    max_bytes: Optional[int] = None,
    force_jpeg: bool = False,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> Optional[PreparedImage]:
    """Trả về ảnh đã chuẩn hóa, hoặc None nếu không đọc được ảnh."""
    try:
        im = Image.open(io.BytesIO(data))
        im.seek(0)
        im.load()
    except Exception:
        return None

    im = _flatten(im)
    if grayscale and im.mode != "L":
        im = im.convert("L")

    width, height = im.size
    if width < 1 or height < 1:
        return None
    scale = min(1.0, max_side / max(width, height))
    if max_width:
        scale = min(scale, max_width / width)
    if max_height:
        scale = min(scale, max_height / height)
    if scale < 1.0:
        im = im.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)

    if not force_jpeg and _is_line_art(im):
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        png = buf.getvalue()
        if not max_bytes or len(png) <= max_bytes:
            return PreparedImage(png, "image/png", "png", im.width, im.height)

    if im.mode not in ("L", "RGB"):
        im = im.convert("RGB")
    quality = 85
    jpeg = _encode_jpeg(im, quality)
    while max_bytes and len(jpeg) > max_bytes:
        if quality > 45:
            quality -= 10
        else:
            new_size = (max(1, int(im.width * 0.8)), max(1, int(im.height * 0.8)))
            if new_size == im.size or min(new_size) < 32:
                break
            im = im.resize(new_size, Image.LANCZOS)
        jpeg = _encode_jpeg(im, quality)
    return PreparedImage(jpeg, "image/jpeg", "jpg", im.width, im.height)
