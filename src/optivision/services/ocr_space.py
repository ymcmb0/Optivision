from __future__ import annotations

import requests
from io import BytesIO
from PIL import Image

from ..config import get_settings

settings = get_settings()

ENDPOINT = "https://api.ocr.space/parse/image"


def run_ocr(img: Image.Image, lang: str = "eng") -> dict:
    """
    Send an in-memory image to OCR.Space and return the parsed JSON.

    Parameters
    ----------
    img   : Pillow Image (RGB, BGR, etc.)
    lang  : ISO-639-1 language code supported by OCR.Space

    Returns
    -------
    dict   OCR.Space JSON response (raises for HTTP/JSON errors)
    """
    # 1. Dump the Pillow image into an in-memory PNG
    buf = BytesIO()
    img.save(buf, format="PNG")          # lossless; change to "JPEG" if smaller is better
    buf.seek(0)

    # 2. Build multipart payload
    files = {
        "file": ("frame.png", buf, "image/png")   # (filename, bytes-like, mime)
    }
    payload = {
        "apikey": settings.ocr_space_api_key,
        "language": lang,
        "OCREngine": 2,                 # 1 = tesseract, 2 = recommended
        "scale": True,
    }

    # 3. Request + error handling
    r = requests.post(ENDPOINT, data=payload, files=files, timeout=30)
    try:
        r.raise_for_status()            # HTTP-level errors
        return r.json()                 # may raise if payload isn’t valid JSON
    finally:
        buf.close()                     # free memory
