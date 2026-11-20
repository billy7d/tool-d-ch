"""Cấu hình theo dòng máy Kindle: độ phân giải màn hình quyết định cỡ ảnh bìa và cỡ ảnh minh họa."""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DeviceProfile:
    key: str
    label: str
    screen_width: int
    screen_height: int
    # Cạnh dài tối đa của ảnh minh họa. Lớn hơn màn hình một chút để còn phóng to trong KOReader,
    # nhưng không giữ ảnh gốc vài nghìn pixel làm máy RAM 256 MB giải mã chậm.
    image_max_side: int
    calibre_profile: str
    description: str


DEVICE_PROFILES: Dict[str, DeviceProfile] = {
    "kindle": DeviceProfile(
        key="kindle",
        label="Kindle 6\" đời cũ – 600×800",
        screen_width=600,
        screen_height=800,
        image_max_side=1200,
        calibre_profile="kindle",
        description="Kindle Keyboard (K3), Kindle 4/5, Kindle Touch, Basic 2014–2019",
    ),
    "kindle_pw": DeviceProfile(
        key="kindle_pw",
        label="Kindle Paperwhite 1/2 – 758×1024",
        screen_width=758,
        screen_height=1024,
        image_max_side=1600,
        calibre_profile="kindle_pw",
        description="Paperwhite 1 (2012), Paperwhite 2 (2013)",
    ),
    "kindle_hd": DeviceProfile(
        key="kindle_hd",
        label="Kindle 300 ppi – 1072×1448",
        screen_width=1072,
        screen_height=1448,
        image_max_side=2000,
        calibre_profile="kindle_pw3",
        description="Paperwhite 3/4, Voyage, Oasis",
    ),
}

DEFAULT_DEVICE = "kindle_pw"


def get_device(key: Optional[str]) -> DeviceProfile:
    return DEVICE_PROFILES.get((key or "").strip().lower(), DEVICE_PROFILES[DEFAULT_DEVICE])


def list_devices() -> List[dict]:
    return [
        {"key": p.key, "label": p.label, "description": p.description,
         "screen_width": p.screen_width, "screen_height": p.screen_height}
        for p in DEVICE_PROFILES.values()
    ]
