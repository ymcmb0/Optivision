## Optivision 📷📝🌦

## Vision + OCR + Weather micro-service built with FastAPI

Detect objects (YOLOv8) → read text (OCR.Space) → classify scene (indoor / outdoor) → append current weather for Lahore.
Everything is packaged so each step can be replaced independently.

---

### 1  Prerequisites

| Tool                | Version(s) tested | Notes                                                     |
| ------------------- | ----------------- | --------------------------------------------------------- |
| **Python**          | 3.10 / 3.11       | Lower versions are unsupported.                           |
| **pip / venv**      | latest            | Comes with Python; upgrade pip after activating the venv. |
| **gcc / Xcode CLT** | optional          | Only if you need to compile native wheels.                |
| **CUDA ≥ 11.8**     | optional          | Required only for GPU inference.                          |

> **NumPy pin** – PyTorch 2.2 wheels are built against NumPy 1.26.
> The dependency list therefore caps NumPy `< 2.0` (with matching SciPy & pandas) until PyTorch 2.3 ships NumPy-2 wheels.

---

### 2  Directory layout

```
optivision/
├─ src/optivision/
│  ├─ api.py                 # FastAPI router
│  ├─ config.py              # Settings / .env loader
│  ├─ models/
│  │  ├─ detector.py         # YOLO interface
│  │  └─ scene.py            # Indoor/Outdoor classifier
│  ├─ services/
│  │  └─ ocr_space.py        # OCR.Space REST helper
│  └─ utils/
│     ├─ images.py
│     └─ weather.py
├─ tests/                    # pytest samples
├─ weights/                  # auto-downloaded YOLO weights
├─ pyproject.toml            # deps & metadata
└─ .env.example              # copy to .env and fill secrets
```

---

### 3  Setup

1. **Create and activate a virtual-environment**

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   # .venv\Scripts\activate         # Windows
   pip install --upgrade pip setuptools wheel
   ```

2. **Install Optivision (editable + dev tools)**

   ```bash
   pip install -e '.[dev]'
   ```

   *If you want runtime-only: `pip install -e .`*

3. **Configure secrets**

   ```dotenv
   # .env  (at repo root)
   OCR_API_KEY=YOUR_OCR_SPACE_KEY
   ```

   *The loader is case-insensitive and also accepts `OCR_SPACE_API_KEY`.*

4. **Install optional extras**

   *For file uploads FastAPI needs `python-multipart` (already declared in the deps list, install manually only if you installed before that change):*

   ```bash
   pip install python-multipart
   ```

---

### 4  Run the service

```bash
uvicorn optivision.api:app \
       --host 0.0.0.0 \
       --port 5002 \
       --reload        # hot-reload during development
```

---

### 5  Testing

```bash
pytest -q
```

The starter test ensures an empty image returns zero detections.
Add more cases in `tests/`.

---

### 6  Troubleshooting

| Error                                        | Fix                                                                                                         |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `ValidationError: ocr_space_api_key missing` | Set `OCR_API_KEY` in `.env` or export it.                                                                   |
| **NumPy 2 vs PyTorch** crash                 | The project pins `numpy<2.0`; reinstall with `pip install "numpy<2.0" "scipy<1.13" "pandas<2.3"` if needed. |
| `python-multipart` required                  | `pip install python-multipart`.                                                                             |

---

### 7  Extending

* **Swap detector** – drop new weights into `weights/` and update `config.py`.
* **Replace OCR provider** – add a new module under `services/` and wire it in `api.py`.
* **Improve scene classifier** – plug in MIT-Places, CLIP, etc., in `models/scene.py`.

---