"""
Indoor / outdoor classifier – MIT Places365 (ResNet-18)
───────────────────────────────────────────────────────
• pre-download the files →  weights/places365/
      ├─ resnet18_places365.pth.tar
      └─ IO_places365.txt
"""

from pathlib import Path
from typing import Literal

import torch
from torchvision import models, transforms
from PIL import Image

# ─── static paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PLACES = ROOT / "weights" / "places365"
MODEL_F = PLACES / "resnet18_places365.pth.tar"
IO_F    = PLACES / "IO_places365.txt"

_missing = [p.name for p in (MODEL_F, IO_F) if not p.exists()]
if _missing:          # pragma: no cover
    raise RuntimeError(
        "❌  Missing MIT Places365 resource(s): "
        + ", ".join(_missing)
        + f"\n   • Put them inside  {PLACES}"
    )

# ─── model ───────────────────────────────────────────────────────────────────
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
_state  = torch.load(MODEL_F, map_location=DEVICE)

MODEL = models.resnet18(num_classes=365)
MODEL.load_state_dict(_state, strict=False)
MODEL.to(DEVICE).eval()

PREP = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ]
)

# ─── IO vector ───────────────────────────────────────────────────────────────
# second token in each line is the IO flag (0/1/2)
IO_VEC = torch.tensor(
    [int(line.split()[-1]) for line in IO_F.read_text().splitlines()],
    dtype=torch.int8,
)  # shape = [365]

# ─── public API ──────────────────────────────────────────────────────────────
def classify_environment(img: Image.Image) -> Literal["indoor", "outdoor"]:
    """
    Return 'indoor' or 'outdoor' (ignores ambiguous *2* classes).

    Strategy: look at the **top-5** predictions; if ≥ 3 vote outdoor (flag == 1)
    the scene is outdoor, else indoor.
    """
    with torch.no_grad():
        t = PREP(img).unsqueeze(0).to(DEVICE)      # [1,3,224,224]
        probs = torch.softmax(MODEL(t), dim=1)[0]  # [365]

        top5 = probs.topk(5).indices               # [5]
        votes = IO_VEC[top5]                       # 0 / 1 / 2

        outdoors = (votes == 1).sum().item()
        indoors  = (votes == 0).sum().item()

        return "outdoor" if outdoors >= 3 else "indoor"
