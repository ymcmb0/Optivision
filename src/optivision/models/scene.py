from pathlib import Path
from typing import Literal

import torch
from torchvision import transforms, models
from PIL import Image

# ─── tiny indoor/outdoor classifier draft ────────────────────────────────────
# Swap out with MIT Places365 if you need real accuracy.

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = models.mobilenet_v2(weights="IMAGENET1K_V1").to(_device).eval()

_preproc = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]
)

def classify_environment(img: Image.Image) -> Literal["indoor", "outdoor"]:
    with torch.no_grad():
        t = _preproc(img).unsqueeze(0).to(_device)
        pred_idx = _model(t)[0].argmax().item()
    # very rough heuristic
    return "outdoor" if pred_idx > 500 else "indoor"
