import io
import shutil
from typing import List, Dict, Any, Tuple
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None


class OCREngine:
    def __init__(self, lang: str = "eng"):
        self.lang = lang
        self.is_available = (pytesseract is not None) and (shutil.which("tesseract") is not None)

    def ocr_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Runs OCR on a PIL Image and returns structured text blocks with bounding box and confidence.
        """
        if not self.is_available:
            return []

        try:
            # Get detailed OCR data with bounding boxes and confidence
            data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)
            
            blocks = []
            n_boxes = len(data["text"])
            
            # Group into paragraphs/lines
            current_block = []
            current_conf = []
            current_bbox = None
            
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])
                
                if not text:
                    continue
                if conf < 0:
                    continue
                    
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                
                # Check line breaks or new paragraphs
                if data["word_num"][i] == 1 and current_block:
                    blocks.append({
                        "text": " ".join(current_block),
                        "confidence": sum(current_conf) / len(current_conf) / 100.0,
                        "bbox": current_bbox,
                    })
                    current_block = []
                    current_conf = []
                    current_bbox = [x, y, x + w, y + h]
                else:
                    if current_bbox is None:
                        current_bbox = [x, y, x + w, y + h]
                    else:
                        current_bbox[0] = min(current_bbox[0], x)
                        current_bbox[1] = min(current_bbox[1], y)
                        current_bbox[2] = max(current_bbox[2], x + w)
                        current_bbox[3] = max(current_bbox[3], y + h)

                current_block.append(text)
                current_conf.append(conf)

            if current_block:
                blocks.append({
                    "text": " ".join(current_block),
                    "confidence": sum(current_conf) / len(current_conf) / 100.0,
                    "bbox": current_bbox,
                })

            return blocks
        except Exception as e:
            print(f"OCR execution warning: {e}")
            return []
