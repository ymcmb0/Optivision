import cv2
import numpy as np
import easyocr
from PIL import Image

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Set your test image path
image_path = r'capture.jpg'

# Load image using OpenCV
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not found. Check the path.")

# Convert to RGB (EasyOCR expects RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Optional preprocessing for better OCR accuracy
def preprocess_image_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return thresh

processed_img = preprocess_image_for_ocr(image_rgb)

# Run OCR on processed image
results = reader.readtext(processed_img)

# Print OCR results
print("\n--- OCR Results ---")
if not results:
    print("No readable text found.")
else:
    for (bbox, text, conf) in results:
        print(f"Text: '{text.strip()}' | Confidence: {conf:.2f}")
