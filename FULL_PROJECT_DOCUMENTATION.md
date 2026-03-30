# Crop Disease Detection - Full Project Documentation

## 1. Project Title
BeejVaidya: Intelligent Crop Disease Detection for Small and Medium Farmers (GenAI-Assisted Explanation)

## 2. Project Description
This project is an end-to-end crop leaf disease detection system that combines a trained deep learning image classifier with a FastAPI backend and a browser-based frontend. Users upload a leaf image, and the system returns disease prediction, confidence, treatment and prevention guidance, and a human-readable explanation.

## 3. Problem Statement
Farmers often lack timely access to agronomy experts and diagnostic tools, which causes delayed disease detection, reduced yield, and potentially excessive pesticide usage. The project aims to provide fast, accessible disease assessment from a single image.

## 4. Solution Overview
The system uses:
- A MobileNetV2 transfer-learning classifier trained on tomato leaf images.
- A FastAPI inference service (`/predict`, `/health`).
- A frontend upload interface (HTML/CSS/JS).
- Safety gates for non-leaf images and low-confidence predictions.
- Rule-based treatment mapping.
- Optional LLM-generated explanation with fallback text.
- CSV logging for inference traceability.

## 5. System Architecture
High-level architecture:
- Frontend sends image via `multipart/form-data` to FastAPI.
- Backend validates image, checks leaf-likeness, runs model inference, applies confidence logic.
- Backend enriches output with treatment/prevention and explanation.
- Backend logs event to CSV and returns JSON response.

### ASCII Architecture Diagram
```text
+---------------------+          HTTP POST /predict          +---------------------------+
|  Web Frontend       | -----------------------------------> |  FastAPI Backend          |
|  (HTML/CSS/JS)      |                                      |  api/main.py              |
+----------+----------+                                      +-------------+-------------+
           |                                                               |
           | GET /health                                                    |
           v                                                               v
+---------------------+                                      +---------------------------+
|  Health Status UI   |                                      |  Inference Pipeline       |
+---------------------+                                      |  1) Image validation      |
                                                             |  2) Green-pixel leaf gate |
                                                             |  3) Keras model predict   |
                                                             |  4) Confidence threshold  |
                                                             |  5) Guidance + explain    |
                                                             +-------------+-------------+
                                                                           |
                               +-------------------------------------------+--------------------------+
                               |                                                                      |
                               v                                                                      v
                  +---------------------------+                                          +--------------------------+
                  | TensorFlow Model (.h5)    |                                          | logs/predictions.csv     |
                  | models/crop_disease_model |                                          | timestamp, class, conf... |
                  +---------------------------+                                          +--------------------------+
                               |
                               v
                  +---------------------------+
                  | utils/treatment_map.py    |
                  | utils/llm_explainer.py    |
                  +---------------------------+
```

## 6. Folder Structure (auto-generated tree)
```text
UNSTOP_TOI_HACKATHON/
|- crop_disease_detection/
|  |- api/
|  |  |- main.py
|  |- dataset/
|  |  |- raw/
|  |  |  |- early_blight/ (1000 files)
|  |  |  |- healthy/ (1591 files)
|  |  |  |- late_blight/ (1909 files)
|  |  |- processed/
|  |  |- split/
|  |  |  |- train/
|  |  |  |  |- early_blight/ (700 files)
|  |  |  |  |- healthy/ (1113 files)
|  |  |  |  |- late_blight/ (1336 files)
|  |  |  |- val/
|  |  |  |  |- early_blight/ (150 files)
|  |  |  |  |- healthy/ (238 files)
|  |  |  |  |- late_blight/ (286 files)
|  |  |  |- test/
|  |  |  |  |- early_blight/ (150 files)
|  |  |  |  |- healthy/ (240 files)
|  |  |  |  |- late_blight/ (287 files)
|  |- frontend/
|  |  |- index.html
|  |  |- script.js
|  |  |- style.css
|  |- logs/
|  |  |- predictions.csv
|  |- models/
|  |  |- crop_disease_model.h5
|  |- results/
|  |- scripts/
|  |  |- split_dataset.py
|  |  |- train_model.py
|  |  |- predict.py
|  |- utils/
|  |  |- treatment_map.py
|  |  |- llm_explainer.py
|  |  |- explain_disease.py
|  |- README.md
|  |- requirements.txt
|  |- setup_project.py
|- GenAI_Crop_Disease_System_Module_Specification.pdf
```

## 7. Technology Stack
### Languages
- Python
- JavaScript
- HTML
- CSS
- Markdown/Text

### Frameworks/Libraries
- FastAPI
- Uvicorn
- TensorFlow / Keras
- NumPy
- Pillow
- `urllib` (for OpenAI HTTP call)

### Declared Dependencies (`requirements.txt`)
- `tensorflow`
- `numpy`
- `pillow`
- `scikit-learn`
- `matplotlib`
- `fastapi`
- `uvicorn`
- `kaggle`

## 8. Machine Learning Model Explanation
Model training (`scripts/train_model.py`) uses transfer learning:
- Base: `MobileNetV2(weights='imagenet', include_top=False)`
- Input: `224x224x3`
- Base frozen (`trainable = False`)
- Custom head:
  - `GlobalAveragePooling2D`
  - `Dense(128, relu)`
  - `Dropout(0.3)`
  - `Dense(3, softmax)`
- Classes: `early_blight`, `healthy`, `late_blight`
- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`
- Epochs: `15`, Batch size: `32`
- Best model checkpoint saved by validation accuracy to `models/crop_disease_model.h5`

## 9. Dataset Information
### Dataset Layout
- Raw data: `dataset/raw/<class>/`
- Split data: `dataset/split/train|val|test/<class>/`

### Current Counts
- Raw total: `4500` images
- Raw class distribution:
  - early_blight: `1000`
  - healthy: `1591`
  - late_blight: `1909`
- Split total: `4500` images
- Split ratio (implemented): `70% train / 15% val / 15% test`

### File Types in Dataset
- Primarily `.JPG/.jpg/.jpeg/.png` image files.

## 10. Backend API Explanation
Backend implementation: `api/main.py`

Main responsibilities:
- Load trained model at startup.
- Provide health status endpoint.
- Accept uploaded images.
- Validate image integrity.
- Run leaf-likeness gate via HSV green pixel ratio.
- Run model inference.
- Apply confidence thresholding.
- Attach treatment + prevention + explanation.
- Log prediction metadata to CSV.

## 11. Frontend Architecture
Frontend implementation: `frontend/index.html`, `frontend/script.js`, `frontend/style.css`

Behavior:
- User selects image file.
- JS shows local preview using `FileReader`.
- On button click, JS sends request to backend using `fetch` and `FormData`.
- Response fields rendered in result card.
- UI handles loading, success, and error states.

## 12. API Endpoints
### `GET /health`
Returns model health and version.

Example:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1.0"
}
```

### `POST /predict`
- Content type: `multipart/form-data`
- Field: `file`
- Returns either:
  - successful disease result with guidance
  - `UNCERTAIN` when non-leaf or low confidence

## 13. Data Flow (end-to-end)
1. User uploads image in browser.
2. Frontend sends file to `POST /predict`.
3. Backend stores temporary file and opens with PIL.
4. Backend computes HSV green ratio to check if image is leaf-like.
5. If fails leaf test, returns `UNCERTAIN` immediately.
6. If leaf-like, backend preprocesses image to `224x224`, normalized `[0,1]`.
7. Model predicts class probabilities.
8. Backend selects top class and confidence.
9. If confidence below threshold, returns `UNCERTAIN`.
10. If confidence passes threshold, backend fetches treatment/prevention mapping.
11. Backend generates explanation (OpenAI if key available, else fallback template).
12. Backend logs prediction row in CSV.
13. Backend returns JSON response to frontend.
14. Frontend displays results to user.

## 14. Prediction Logic
Prediction flow in `api/main.py`:
- `preds = model.predict(input_tensor)[0]`
- `idx = argmax(preds)`
- `disease = CLASS_NAMES[idx]`
- `confidence = preds[idx]`
- Returns disease only if confidence and leaf gates pass.

## 15. Confidence Evaluation Logic
- `CONFIDENCE_THRESHOLD = 0.75`
- Confidence bands:
  - `>= 0.90` -> `high`
  - `>= 0.75` and `< 0.90` -> `medium`
  - `< 0.75` -> `low`
- If `< 0.75`, response is:
  - `prediction: "UNCERTAIN"`
  - message: `"Image unclear or disease not recognized"`

## 16. Leaf Detection Logic
Leaf gate (`green_pixel_ratio`) converts image to HSV and marks pixels as green if:
- Hue between `25` and `90`
- Saturation >= `40`
- Value >= `40`

Thresholds:
- API inference gate: `LEAF_GREEN_THRESHOLD = 0.05`
- CLI script (`scripts/predict.py`) gate: `0.10`

If ratio is below threshold, API returns `UNCERTAIN` with non-leaf message.

## 17. Treatment Mapping System
Defined in `utils/treatment_map.py` as static mapping:
- `early_blight` -> treatment + prevention text
- `late_blight` -> treatment + prevention text
- `healthy` -> maintenance guidance

This ensures deterministic, low-latency guidance even without external services.

## 18. Explanation Generator
Module: `utils/llm_explainer.py`

Logic:
- If `OPENAI_API_KEY` not set: return fallback explanation template.
- If set: call OpenAI Chat Completions endpoint (`/v1/chat/completions`) using:
  - model from `OPENAI_MODEL` (default `gpt-4o-mini`)
  - system + user prompt containing prediction context
- If API/network/JSON error: fallback explanation is returned.

Note: Uses direct HTTP via `urllib`, not OpenAI SDK.

## 19. Logging System
File: `logs/predictions.csv`

Columns:
- `timestamp`
- `image_name`
- `prediction`
- `confidence`
- `confidence_level`

Behavior:
- CSV header auto-created on startup if file absent.
- A row is appended for every prediction (including uncertain cases).

## 20. Health Monitoring Endpoint
`GET /health` provides lightweight service observability:
- model load status
- API status (`ok`/`error`)
- model version string (`v1.0`)

Useful for service checks, orchestration probes, and frontend readiness checks.

## 21. Performance Information
Available from implementation:
- Per-request processing time is computed and returned as `processing_time_ms`.
- Model is loaded once on API startup (avoids reload per request).
- Image preprocessing includes downscaling and fixed-size resize (`224x224`) to keep inference cost bounded.

Current repository does not include dedicated benchmarking or load-test scripts.

## 22. Security and Safety Handling
Implemented safety features:
- Invalid image handling: catches `UnidentifiedImageError`, returns HTTP `400`.
- Non-leaf detection gate to avoid misleading crop diagnosis on unrelated images.
- Low-confidence abstention (`UNCERTAIN`) to reduce overconfident false guidance.
- Temporary upload file is removed in `finally` block.

Current security gaps:
- CORS is fully open (`allow_origins=['*']`).
- No authentication/authorization.
- No rate limiting or abuse controls.
- No file size limit enforcement in endpoint.

## 23. How to Run the Project
### Backend
```bash
cd crop_disease_detection
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend
```bash
cd crop_disease_detection/frontend
python -m http.server 5500
```
Open: `http://127.0.0.1:5500`

## 24. Installation Instructions
1. Clone or open repository.
2. Create virtual environment using Python 3.10.
3. Install dependencies from `requirements.txt` inside `.venv`.
4. Ensure model file exists at `models/crop_disease_model.h5`.
5. (Optional) set environment variables:
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL`
6. Start backend and frontend servers.

### Troubleshooting
- `bad interpreter: .../venv/bin/python3.10: No such file or directory`
  - Recreate the virtual environment in the current path.
- `No matching distribution found for tensorflow`
  - Use Python 3.10, then reinstall dependencies.
- `ModuleNotFoundError: No module named 'tensorflow'`
  - Activate `.venv` and run backend with `python -m uvicorn ...`.

## 25. Future Improvements
- Add automated test suite (unit + integration + API contract tests).
- Add evaluation scripts (confusion matrix, per-class precision/recall/F1).
- Add model versioning and model registry metadata.
- Add authentication and API key-based access.
- Restrict CORS and add rate limiting.
- Add input file size/type constraints and malware scanning hooks.
- Add structured logging and monitoring dashboards.
- Add Docker/Compose deployment and CI pipeline.
- Add multilingual explanation outputs for farmer usability.
- Expand to more crops/diseases and active learning retraining loop.

---

## Example API Request and Response
### Request (cURL)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf.jpg"
```

### Example Success Response
```json
{
  "prediction": "late_blight",
  "confidence": 0.9821,
  "confidence_level": "high",
  "treatment": "Remove infected plants immediately and apply fungicide.",
  "prevention": "Ensure proper air circulation and avoid excess moisture.",
  "explanation": "Disease detected: Late Blight...",
  "timestamp": "2026-03-08T12:30:00",
  "model_version": "v1.0",
  "processing_time_ms": 126
}
```

### Example Uncertain Response
```json
{
  "prediction": "UNCERTAIN",
  "confidence": 0.6412,
  "confidence_level": "low",
  "message": "Image unclear or disease not recognized",
  "timestamp": "2026-03-08T12:31:10",
  "model_version": "v1.0",
  "processing_time_ms": 118
}
```

---

## Step-by-Step System Workflow
1. Backend starts and loads model from `models/crop_disease_model.h5`.
2. Frontend loads and waits for image selection.
3. User uploads leaf image.
4. Frontend previews image and sends it to `/predict`.
5. Backend validates image.
6. Backend runs leaf gate (green HSV ratio).
7. If non-leaf, backend returns `UNCERTAIN`.
8. Else backend preprocesses image and runs CNN inference.
9. Backend evaluates top probability against confidence threshold.
10. If low confidence, backend returns `UNCERTAIN`.
11. If confident, backend maps treatment/prevention.
12. Backend generates explanation (LLM or fallback).
13. Backend writes prediction log to CSV.
14. Frontend displays final diagnosis and guidance.

---

## Repository Scan Summary (derived)
- Workspace root files: `43,375`
- Workspace root directories: `5,381`
- Main project folder: `crop_disease_detection/`
- Dominant file type: image files (`.jpg/.jpeg/.png`) due to dataset.
- Core source files: Python backend/scripts/utils + JS/HTML/CSS frontend.
