#  ────────────────────────────────────────────────────────────────────────────
#  utils / text_narrator.py
#  Craft one natural-sounding sentence that summarises what the system sees.
#  ────────────────────────────────────────────────────────────────────────────
"""
craft_sentence(
    objects : list[str] | set[str],
    ocr_text: str | None,
    env     : Literal["indoor", "outdoor"],
    weather : str | None
) -> str
"""

from __future__ import annotations
from typing import Literal

__all__ = ["craft_sentence"]

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
_VOWELS = set("aeiou")


def _articlise(noun: str) -> str:
    """
    Return the noun with a suitable indefinite article:
        chair   -> "a chair"
        apple   -> "an apple"
        books   -> "books"        (plural → no article)
    """
    noun = noun.strip()
    return noun if noun.endswith("s") else (
        f"{'an' if noun[0].lower() in _VOWELS else 'a'} {noun}"
    )


def _list_with_commas(items: list[str]) -> str:
    """
    Oxford-comma list:
        ["a chair"]                       -> "a chair"
        ["a chair", "a table"]            -> "a chair and a table"
        ["a chair", "a table", "a lamp"]  -> "a chair, a table and a lamp"
    """
    if not items:
        return "nothing in particular"
    if len(items) == 1:
        return items[0]
    head, tail = ", ".join(items[:-1]), items[-1]
    conjunction = "and" if len(items) == 2 else "and"
    return f"{head} {conjunction} {tail}"


# --------------------------------------------------------------------------- #
# public                                                                      #
# --------------------------------------------------------------------------- #
def craft_sentence(
    objects: list[str] | set[str],
    ocr_text: str | None,
    env: Literal["indoor", "outdoor"],
    weather: str | None,
) -> str:
    """
    Build a friendly one-liner for TTS / UI, always in the same order:
        • objects
        • OCR (if any)
        • scene type
        • weather (if outdoor)
    """
    # ---------- objects ---------------------------------------------------- #
    obj_words = sorted(objects)
    obj_phrase = f"I can see {_list_with_commas([_articlise(o) for o in obj_words])}."

    # ---------- OCR -------------------------------------------------------- #
    ocr_phrase = ""
    print(f"🔍 OCR text: {ocr_text!r}")
    if ocr_text:
        cleaned = ocr_text.strip()
        # Short snippets are quoted; long ones are paraphrased
        if len(cleaned.split()) <= 12:
            ocr_phrase = f' The visible text says: "{cleaned}".'
        else:
            ocr_phrase = " There is some printed text in view."

    # ---------- scene & weather ------------------------------------------- #
    scene_phrase = f" It looks like an {env} scene."
    weather_phrase = f" Weather feels {weather.lower()}." if weather else ""

    return (obj_phrase + ocr_phrase + scene_phrase + weather_phrase).strip()
