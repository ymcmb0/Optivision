# src/optivision/utils/text_narrator.py
"""
Craft a single, human-friendly sentence that summarises the vision pipeline.

Signature
─────────
    craft_sentence(
        objects: list[str] | set[str],
        ocr_text: str | None,
        env:      Literal["indoor", "outdoor"],
        weather:  str | None
    ) -> str
"""

from __future__ import annotations
from typing import Literal

__all__ = ["craft_sentence"]


def _join_items(items: list[str]) -> str:
    """Return 'a chair', 'a chair and a table', 'a chair, a table and a lamp'."""
    if not items:
        return "nothing in particular"
    if len(items) == 1:
        return f"a {items[0]}"
    if len(items) == 2:
        return f"a {items[0]} and a {items[1]}"
    head, tail = ", ".join(items[:-1]), items[-1]
    return f"{head}, and a {tail}"


def craft_sentence(
    objects: list[str] | set[str],
    ocr_text: str | None,
    env: Literal["indoor", "outdoor"],
    weather: str | None,
) -> str:
    """Compose a natural-sounding description for TTS / UI."""
    # 1. Objects
    obj_phrase = f"I can see {_join_items(sorted(objects))}."

    # 2. Scene type
    scene_phrase = f"It looks like an {env} scene."

    # 3. Weather (optional)
    weather_phrase = f" Weather feels {weather.lower()}." if weather else ""

    # 4. OCR (optional) – trim and quote if short, otherwise summarise
    ocr_phrase = ""
    if ocr_text:
        text = ocr_text.strip()
        if len(text.split()) <= 12:
            ocr_phrase = f' The text reads “{text}.”'
        else:
            ocr_phrase = " There’s some printed text in view."

    return (obj_phrase + " " + scene_phrase + weather_phrase + ocr_phrase).strip()
