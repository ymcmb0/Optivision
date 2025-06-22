"""
Cloud-only OCR helper (OCR.Space)
─────────────────────────────────
run_ocr(img: PIL.Image) ➜ str | None
"""

from __future__ import annotations
import io, time, hashlib, threading
from typing import Optional

import requests
from PIL import Image

from ..config import get_settings

# ─── settings & constants ────────────────────────────────────────────────────
settings   = get_settings()
API_KEY: str | None = getattr(settings, "ocr_space_api_key", None)
ENDPOINT   = "https://api.ocr.space/parse/image"

# ‣ Throttle: at most N calls every WINDOW seconds
MAX_CALLS  = 30           # free-tier:  FREE plan =  3 /min  (≈180 /hour)
WINDOW     = 60.0         # premium:   LIMITED plan = 30/min
#   tweak those two numbers to suit your plan

# --------------------------------------------------------------------------- #
# internals                                                                   #
# --------------------------------------------------------------------------- #
_lock             = threading.Lock()
_calls_this_cycle = 0
_cycle_start      = time.monotonic()
_cache: dict[str, str] = {}          # image-hash → parsed text  (tiny LRU)

def _hash(img: Image.Image) -> str:
    """SHA-1 of JPEG bytes – avoids re-submitting identical frames."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=50, optimize=True)
    return hashlib.sha1(buf.getvalue()).hexdigest()


def _encode(img: Image.Image) -> tuple[dict, dict]:
    # down-size to width ≤ 800 px (OCR.Space recommendation)
    w, h = img.size
    if w > 800:
        img = img.resize((800, int(h * 800 / w)), Image.BICUBIC)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    buf.seek(0)

    data = {
        "apikey":    API_KEY,
        "language":  "eng",
        "scale":     "true",   # sharpen a bit in the cloud
    }
    files = {"file": ("frame.jpg", buf, "image/jpeg")}
    return data, files


def _rate_limit() -> bool:
    """
    Return True if we are allowed to call the API right now,
    otherwise sleep until the window resets and then return True.
    """
    global _calls_this_cycle, _cycle_start

    with _lock:
        now = time.monotonic()
        elapsed = now - _cycle_start

        if elapsed >= WINDOW:
            # new window
            _cycle_start      = now
            _calls_this_cycle = 0

        if _calls_this_cycle < MAX_CALLS:
            _calls_this_cycle += 1
            return True

        # hit the ceiling – wait for window reset
        wait = WINDOW - elapsed
        print(f"⏳  OCR.Space throttle: sleeping {wait:0.1f}s …")
    time.sleep(wait)
    return _rate_limit()      # recurse once – cheap


def _request(data: dict, files: dict, timeout=60) -> Optional[str]:
    try:
        r = requests.post(ENDPOINT, data=data, files=files, timeout=timeout)
        r.raise_for_status()
        txt = (
            r.json()
             .get("ParsedResults", [{}])[0]
             .get("ParsedText", "")
        ).strip()
        return txt or None
    except Exception as e:
        print(f"⚠️  OCR.Space error: {e}")
        return None


# --------------------------------------------------------------------------- #
# public                                                                      #
# --------------------------------------------------------------------------- #
def run_ocr(img: Image.Image) -> Optional[str]:
    """
    • Requires `OCR_SPACE_API_KEY` (or `OCR_API_KEY`) in env/.env  
    • Observes a simple token bucket so the free tier isn’t exhausted.  
    • Caches identical frames by SHA-1 hash (memory-only, tiny).
    """
    if not API_KEY:
        print("⚠️  run_ocr: No OCR.Space API key configured.")
        return None

    key = _hash(img)
    if key in _cache:
        return _cache[key]

    _rate_limit()                              # blocks if we’re too fast
    data, files = _encode(img)

    txt = _request(data, files)
    if txt:
        # keep cache modest (e.g. last 150 entries)
        if len(_cache) > 150:
            _cache.pop(next(iter(_cache)))
        _cache[key] = txt
    return txt
