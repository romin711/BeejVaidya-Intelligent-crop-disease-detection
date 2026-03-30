# BeejVaidya

**Beej** (Seed) **Vaidya** (Doctor) — Intelligent crop disease detection for farmers.

End-to-end crop disease detection project with:
- A FastAPI backend for inference
- A TensorFlow/Keras image classification model
- Utility modules for treatment + explanation generation
- A simple HTML/CSS/JavaScript frontend

The current model predicts three classes:
- `early_blight`
- `healthy`
- `late_blight`

## Tech Stack

- Python 3.10 (recommended for TensorFlow compatibility)
- FastAPI + Uvicorn
- TensorFlow / Keras (MobileNetV2 transfer learning)
- NumPy
- Pillow
- Scikit-learn (dependency listed)
- Matplotlib (dependency listed)
- Vanilla frontend: HTML, CSS, JavaScript (`fetch`, `FormData`)
- Optional LLM explanation via OpenAI Chat Completions API

## Project Structure

```text
crop_disease_detection/
├── api/
│   └── main.py                  # FastAPI app, model load, prediction logic, logging
├── dataset/
│   ├── raw/
│   │   ├── early_blight/
│   │   ├── healthy/
│   │   └── late_blight/
│   ├── processed/
│   └── split/
│       ├── train/
│       ├── val/
│       └── test/
├── frontend/
│   ├── index.html               # Upload UI + result display
│   ├── script.js                # Calls backend /predict endpoint
│   └── style.css                # Styling
├── logs/
│   └── predictions.csv          # Inference logs
├── models/
│   └── crop_disease_model.h5    # Trained Keras model artifact
├── results/
│   ├── logs/
│   └── reports/
├── scripts/
│   ├── split_dataset.py         # 70/15/15 split from dataset/raw -> dataset/split
│   ├── train_model.py           # Train MobileNetV2 classifier + save best model
│   └── predict.py               # CLI single-image prediction
├── utils/
│   ├── treatment_map.py         # Disease -> treatment/prevention mapping
│   ├── llm_explainer.py         # Optional OpenAI-based explanation with fallback
│   └── explain_disease.py       # Rule-based explanation helper
├── requirements.txt
├── setup_project.py             # Scaffold generator for directories/files
└── README.md
```

## Backend API

Base URL (local): `http://127.0.0.1:8000`

### `GET /health`
Health check + model load status.

Example response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1.0"
}
```

### `POST /predict`
Accepts one uploaded image file (`multipart/form-data`, field name: `file`).

Core behavior in `api/main.py`:
- Validates uploaded image using Pillow
- Performs a leaf-likeness check using green pixel ratio in HSV
- Runs model inference (`models/crop_disease_model.h5`)
- Applies uncertainty handling:
  - If image is not leaf-like -> `UNCERTAIN`
  - If confidence `< 0.75` -> `UNCERTAIN`
- Adds treatment + prevention guidance from `utils/treatment_map.py`
- Generates explanation via `utils/llm_explainer.py`
  - Uses OpenAI API if `OPENAI_API_KEY` is set
  - Falls back to built-in template otherwise
- Logs each prediction to `logs/predictions.csv`

Example `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

## Model Details

Implemented in `scripts/train_model.py`:
- Base model: `MobileNetV2` with ImageNet weights, top removed
- Input size: `224x224x3`
- Head: `GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.3) -> Dense(3, softmax)`
- Training setup:
  - Batch size: `32`
  - Epochs: `15`
  - Loss: `categorical_crossentropy`
  - Optimizer: `Adam`
  - Metric: `accuracy`
- Augmentation on train set:
  - Rescale, rotation, zoom, horizontal flip
- Checkpoint:
  - Saves best model by validation accuracy to `models/crop_disease_model.h5`

Inference logic uses:
- Class order: `early_blight`, `healthy`, `late_blight`
- Confidence threshold: `0.75`
- Leaf green ratio guard: `0.05` in API (`0.10` in CLI script)

## Dataset Snapshot (Current Repository)

`dataset/raw` currently contains:
- `early_blight`: 1000 images
- `healthy`: 1591 images
- `late_blight`: 1909 images

`dataset/split` currently contains:
- Train: 700 / 1113 / 1336
- Val: 150 / 238 / 286
- Test: 150 / 240 / 287

(ordered as `early_blight / healthy / late_blight`)

## End-to-End Workflow

1. Prepare class-wise images in `dataset/raw/<class_name>/`.
2. Split the dataset into train/val/test.
3. Train model and save best checkpoint to `models/`.
4. Start FastAPI backend, which loads model at startup.
5. Upload leaf image from frontend (or call API directly).
6. Receive prediction, confidence, guidance, explanation, and metadata.
7. Persist prediction log in `logs/predictions.csv`.

## Setup and Run

### 1. Create a Python 3.10 virtual environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Why Python 3.10: TensorFlow wheels are not available for Python 3.14 in this project setup.

### 2. Configure environment variables
Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
EOF
```

`utils/llm_explainer.py` reads this `.env` file automatically.

### 3. (Optional) Regenerate scaffold
```bash
python setup_project.py
```

### 4. Split dataset
```bash
python scripts/split_dataset.py
```

### 5. Train model
```bash
python scripts/train_model.py
```

### 6. Run backend API
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 7. Run frontend
From another terminal:
```bash
cd frontend
python -m http.server 5500
```
Open `http://127.0.0.1:5500` in browser.

## Troubleshooting

- `bad interpreter: .../venv/bin/python3.10: No such file or directory`
  - Cause: virtual environment was created in a different path and then moved.
  - Fix: delete that environment and recreate it in the current project path.

- `No matching distribution found for tensorflow`
  - Cause: running `pip install tensorflow` with an unsupported Python version (commonly 3.14).
  - Fix: use Python 3.10 and reinstall dependencies inside `.venv`.

- `ModuleNotFoundError: No module named 'tensorflow'`
  - Cause: backend started with a Python interpreter that does not have TensorFlow installed.
  - Fix: activate `.venv` first, then run backend using `python -m uvicorn ...`.

## Environment Variables

Variables used by `utils/llm_explainer.py` (loaded from project-root `.env`):
- `OPENAI_API_KEY` -> enables OpenAI-generated explanations
- `OPENAI_MODEL` -> defaults to `gpt-4o-mini`

If `OPENAI_API_KEY` is missing or request fails, fallback explanation text is used.

## Frontend Notes

`frontend/script.js` currently points to:
- `const API_URL = "http://127.0.0.1:8000/predict"`

If your backend runs on a different host/port, update this constant.

## Notes

- The repository currently includes training, splitting, and single-image prediction scripts.
- Dataset validation and evaluation-report scripts can be added later if needed.
