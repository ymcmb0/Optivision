from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
from dotenv import load_dotenv
import requests, os
import torch
from torchvision import transforms, models
load_dotenv()
app = Flask(__name__)
model = YOLO("yolov8s.pt")
ocr_api_key = os.getenv("ocr_api_key")
print(ocr_api_key)
# 🔍 OCR.Space API
def compress_image(image_pil, output_path="compressed_image.jpg", quality=40):
    img = image_pil.resize((image_pil.width // 2, image_pil.height // 2))
    img = img.convert("RGB")
    img.save(output_path, format="JPEG", optimize=True, quality=quality)
    return output_path

def ocr_space_file(filename, overlay=False, api_key=ocr_api_key, language='eng'):
    payload = {
        'isOverlayRequired': overlay,
        'apikey': api_key,
        'language': language,
    }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload)
    return r.json()

# 🧠 Indoor/Outdoor Scene Classifier
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.eval()

scene_labels = {0: "indoor", 1: "outdoor"}  # simplistic binary classifier for this use

def is_outdoor_scene(image_pil):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image_pil).unsqueeze(0)
    with torch.no_grad():
        output = mobilenet(input_tensor)
    pred_idx = output[0].argmax().item()
    # crude assumption: label index < 500 → indoor, else outdoor (not perfect)
    return "outdoor" if pred_idx > 500 else "indoor"

# 🌦️ Weather API (OpenWeatherMap)
def get_weather_condition(city="Lahore"):
    try:
            # Example: Lahore, Pakistan
        latitude = 31.5204
        longitude = 74.3587
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'current': 'weather_code',
            'timezone': 'auto'
        }

        # Weather code to description map (simplified)
        weather_code_map = {
            0: 'Clear sky',
            1: 'Mainly clear',
            2: 'Partly cloudy',
            3: 'Overcast',
            45: 'Fog',
            48: 'Depositing rime fog',
            51: 'Light drizzle',
            53: 'Moderate drizzle',
            55: 'Dense drizzle',
            61: 'Light rain',
            63: 'Moderate rain',
            65: 'Heavy rain',
            71: 'Light snow',
            73: 'Moderate snow',
            75: 'Heavy snow',
            95: 'Thunderstorm',
            96: 'Thunderstorm with hail'
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            weather_code = data['current']['weather_code']
            weather_type = weather_code_map.get(weather_code, "Unknown")
            return weather_type
        else:
            return "Couldn't read weather properly!"
    except:
        return "unknown"

@app.route('/inference', methods=['POST'])
def detect_and_read():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

    image_np = np.array(image)

    # ✅ 1. Object Detection
    results = model(image)
    boxes = results[0].boxes
    detection_output = []
    detected_classes = []

    for box in boxes:
        cls = results[0].names[int(box.cls[0])]
        conf = float(box.conf[0])
        coords = [float(x) for x in box.xyxy[0].tolist()]
        detection_output.append({
            'class': cls,
            'confidence': conf,
            'bbox': coords
        })
        detected_classes.append(cls)

    # ✅ 2. OCR
    compressed_path = compress_image(image)
    result = ocr_space_file(filename=compressed_path)

    paragraph_text = "no readable text"
    ocr_output = []

    try:
        extracted_text = result['ParsedResults'][0]['ParsedText']
        paragraph_text = extracted_text.strip()
        ocr_output.append({
            'text': paragraph_text,
            'confidence': 1.0,
            'bbox': []
        })
    except Exception as e:
        print("OCR.Space Error:", e)
        print("Response:", result)

    #✅ 3. Scene classification
    environment = is_outdoor_scene(image)
    weather = get_weather_condition() if environment == "outdoor" else None

   # ✅ 4. Summary
    object_list = sorted(set(detected_classes))
    object_text = ', '.join(object_list) if object_list else "no objects"
    paragraph_output = paragraph_text if paragraph_text else "no readable text"

    message = f"I can see these objects: {object_text}. Text reads: {paragraph_output}."
    message= "Hi "
    if environment == "outdoor":
        message += f" It's an outdoor scene. Weather seems {weather}."
    else:
        message += f" It's an indoor scene. "

    return jsonify({
        'summary': message,
        'detections': detection_output,
        'ocr_text': ocr_output,
        'environment': environment,
        'weather': weather
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
