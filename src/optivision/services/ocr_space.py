import io
import logging
from typing import Dict, Any
import requests
from PIL import Image

from ..config import get_settings
from ..utils.images import compress_image

log = logging.getLogger(__name__)
settings = get_settings()

def run_ocr(image: Image.Image, lang: str = "eng") -> Dict[str, Any]:
    """Call OCR.Space API and return parsed JSON."""
    img_buf = io.BytesIO()
    compress_image(image, img_buf)                 # write JPEG into buffer

    payload = {
        "isOverlayRequired": False,
        "apikey": settings.ocr_space_api_key,
        "language": lang,
    }
    files = {"image.jpg": img_buf.getvalue()}
    r = requests.post("https://api.ocr.space/parse/image",
                      files=files, data=payload, timeout=15)
    r.raise_for_status()
    return r.json()
