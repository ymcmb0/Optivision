from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np

from .models.detector import detect_objects
from .models.scene import classify_environment
from .services.ocr_space import run_ocr
from .utils.weather import current_weather
from .config import get_settings

app = FastAPI(title="Optivision Inference API")
settings = get_settings()

@app.post("/inference")
async def inference(image: UploadFile = File(...)):
    try:
        payload = await image.read()
        img = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid image file")

    # 1️⃣ object detection
    detections = detect_objects(img)
    object_list = sorted({d.class_name for d in detections}) or ["no objects"]

    # 2️⃣ OCR
    ocr_json = run_ocr(img)
    try:
        paragraph = ocr_json["ParsedResults"][0]["ParsedText"].strip()
    except Exception:
        paragraph = ""
    ocr_text = paragraph or "no readable text"

    # 3️⃣ scene + weather
    env = classify_environment(img)
    weather = (
        current_weather(31.5204, 74.3587)     # Lahore default
        if env == "outdoor" else None
    )

    # 4️⃣ summary
    summary = (
        f"I can see these objects: {', '.join(object_list)}. "
        f"Text reads: {ocr_text}. "
        f"It's an {env} scene."
        + (f" Weather seems {weather}." if weather else "")
    )

    return JSONResponse(
        content={
            "summary": summary,
            "detections": [d.dict() for d in detections],
            "ocr_text": ocr_text,
            "environment": env,
            "weather": weather,
        }
    )
