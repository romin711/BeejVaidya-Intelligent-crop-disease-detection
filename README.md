<div align="center">

<h1>🌿 BeejVaidya</h1>

<p><strong>बीज वैद्य</strong> &nbsp;|&nbsp; <em>Seed Doctor</em></p>

<p><em>Intelligent Crop Disease Detection for Farmers</em></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Model-MobileNetV2-orange?style=flat-square" alt="MobileNetV2"/>
  <img src="https://img.shields.io/badge/Classes-3-blue?style=flat-square" alt="3 Classes"/>
  <img src="https://img.shields.io/badge/Input-224×224-purple?style=flat-square" alt="Input Size"/>
  <img src="https://img.shields.io/badge/Confidence%20Threshold-75%25-red?style=flat-square" alt="Confidence Threshold"/>
</p>

---

> 🌱 **Beej** (Seed) · 🩺 **Vaidya** (Doctor) — AI-powered plant health diagnostics bringing modern ML to the farm.

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [✅ Latest Validation](#-latest-validation)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Backend API](#-backend-api)
- [🧠 Model Details](#-model-details)
- [📊 Dataset Snapshot](#-dataset-snapshot)
- [🔄 End-to-End Workflow](#-end-to-end-workflow)
- [⚙️ Setup and Run](#️-setup-and-run)
- [🔧 Troubleshooting](#-troubleshooting)
- [🌐 Environment Variables](#-environment-variables)
- [🖥️ Frontend Notes](#️-frontend-notes)
- [📝 Notes](#-notes)

---

## ✨ Overview

**BeejVaidya** is a full-stack, end-to-end crop disease detection system that enables farmers to quickly diagnose plant health issues by simply uploading a leaf image.

<table>
<tr>
<td>🔬 <strong>AI Inference</strong></td>
<td>FastAPI backend with TensorFlow/Keras MobileNetV2 model</td>
</tr>
<tr>
<td>🌿 <strong>Disease Classes</strong></td>
<td><code>early_blight</code> · <code>healthy</code> · <code>late_blight</code></td>
</tr>
<tr>
<td>💊 <strong>Treatment Guidance</strong></td>
<td>Disease-specific treatment & prevention recommendations</td>
</tr>
<tr>
<td>🤖 <strong>LLM Explanations</strong></td>
<td>Optional OpenAI-powered explanations with rule-based fallback</td>
</tr>
<tr>
<td>🖥️ <strong>Frontend</strong></td>
<td>Clean HTML/CSS/JS upload UI with result display</td>
</tr>
</table>

---

## ✅ Latest Validation

Current verified status from the latest local run:

- Backend startup completed successfully (`/health` and `/predict` responded `200 OK`)
- Frontend static files loaded correctly (`index.html`, `style.css`, `script.js`)
- Late blight diagnostic check passed on 5/5 test samples
- Model file `models/crop_disease_model.h5` loaded correctly at startup

### Late Blight Sanity Check (recent run)

| Metric | Result |
|--------|--------|
| Samples tested | 5 |
| Correct `late_blight` predictions | 5 |
| Pass rate | 100% |

---

---

## 🛠️ Tech Stack

<table>
<thead>
<tr><th>Layer</th><th>Technology</th><th>Purpose</th></tr>
</thead>
<tbody>
<tr><td>🐍 <strong>Runtime</strong></td><td>Python 3.10</td><td>Recommended for TensorFlow compatibility</td></tr>
<tr><td>⚡ <strong>Backend</strong></td><td>FastAPI + Uvicorn</td><td>High-performance async API server</td></tr>
<tr><td>🧠 <strong>ML Framework</strong></td><td>TensorFlow / Keras</td><td>MobileNetV2 transfer learning</td></tr>
<tr><td>🔢 <strong>Numerics</strong></td><td>NumPy</td><td>Array operations & preprocessing</td></tr>
<tr><td>🖼️ <strong>Imaging</strong></td><td>Pillow</td><td>Image validation & processing</td></tr>
<tr><td>📐 <strong>ML Utilities</strong></td><td>Scikit-learn</td><td>Dataset utilities (listed dependency)</td></tr>
<tr><td>📊 <strong>Visualization</strong></td><td>Matplotlib</td><td>Plots & charts (listed dependency)</td></tr>
<tr><td>🌐 <strong>Frontend</strong></td><td>HTML · CSS · JavaScript</td><td>Vanilla UI with <code>fetch</code> / <code>FormData</code></td></tr>
<tr><td>🤖 <strong>LLM</strong></td><td>OpenAI Chat Completions API</td><td>Optional AI-generated explanations</td></tr>
</tbody>
</table>

---

## 📁 Project Structure

```text
crop_disease_detection/
├── 📂 api/
│   └── main.py                  # FastAPI app, model load, prediction logic, logging
├── 📂 dataset/
│   ├── raw/
│   │   ├── early_blight/
│   │   ├── healthy/
│   │   └── late_blight/
│   ├── processed/
│   └── split/
│       ├── train/
│       ├── val/
│       └── test/
├── 📂 frontend/
│   ├── index.html               # Upload UI + result display
│   ├── script.js                # Calls backend /predict endpoint
│   └── style.css                # Styling
├── 📂 logs/
│   └── predictions.csv          # Inference logs
├── 📂 models/
│   └── crop_disease_model.h5    # Trained Keras model artifact
├── 📂 results/
│   ├── logs/
│   └── reports/
├── 📂 scripts/
│   ├── split_dataset.py         # 70/15/15 split from dataset/raw -> dataset/split
│   ├── train_model.py           # Train MobileNetV2 classifier + save best model
│   └── predict.py               # CLI single-image prediction
├── 📂 utils/
│   ├── treatment_map.py         # Disease -> treatment/prevention mapping
│   ├── llm_explainer.py         # Optional OpenAI-based explanation with fallback
│   └── explain_disease.py       # Rule-based explanation helper
├── 📄 requirements.txt
├── 🔧 setup_project.py          # Scaffold generator for directories/files
└── 📖 README.md
```

---

## 🚀 Backend API

**Base URL (local):** `http://127.0.0.1:8000`

### `GET /health`

> Health check + model load status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1.0"
}
```

---

### `POST /predict`

> Accepts one uploaded image file (`multipart/form-data`, field name: `file`).

**Processing Pipeline:**

```
Upload Image
     │
     ▼
🖼️  Validate with Pillow
     │
     ▼
🌿  Leaf-likeness Check (HSV green pixel ratio)
     │
     ▼
🧠  Model Inference (crop_disease_model.h5)
     │
     ▼
📊  Confidence Check (threshold: 0.75)
     │                    │
  ≥ 0.75              < 0.75 or not leaf
     │                    │
     ▼                    ▼
🏷️  Predicted Class    ⚠️  UNCERTAIN
     │
     ▼
💊  Treatment + Prevention (treatment_map.py)
     │
     ▼
🤖  Explanation (OpenAI API or rule-based fallback)
     │
     ▼
📝  Log to predictions.csv
```

**Example `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

---

## 🧠 Model Details

Implemented in `scripts/train_model.py`:

| Parameter | Value |
|-----------|-------|
| **Base Model** | MobileNetV2 (ImageNet weights, top removed) |
| **Input Size** | `224 × 224 × 3` |
| **Head Architecture** | `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(3, softmax)` |
| **Batch Size** | `32` |
| **Epochs** | `15` |
| **Loss Function** | `categorical_crossentropy` |
| **Optimizer** | `Adam` |
| **Metric** | `accuracy` |
| **Confidence Threshold** | `0.75` |
| **Leaf Green Ratio Guard** | `0.05` (API) · `0.10` (CLI) |

**Data Augmentation (train set):**
- ↻ Rotation &nbsp;·&nbsp; 🔍 Zoom &nbsp;·&nbsp; ↔️ Horizontal flip &nbsp;·&nbsp; 📐 Rescale

> **Checkpoint:** Saves best model by validation accuracy to `models/crop_disease_model.h5`

---

## 📊 Dataset Snapshot

### Raw Dataset (`dataset/raw`)

| Class | Images |
|-------|--------|
| 🟡 `early_blight` | 1,000 |
| 🟢 `healthy` | 1,591 |
| 🔴 `late_blight` | 1,909 |
| **Total** | **4,500** |

### Split Dataset (`dataset/split`) — 70 / 15 / 15

| Split | early_blight | healthy | late_blight | Total |
|-------|-------------|---------|-------------|-------|
| 🏋️ Train | 700 | 1,113 | 1,336 | 3,149 |
| ✅ Val | 150 | 238 | 286 | 674 |
| 🧪 Test | 150 | 240 | 287 | 677 |

---

## 🔄 End-to-End Workflow

```
Step 1 ──▶  📁  Prepare class-wise images in dataset/raw/<class_name>/
Step 2 ──▶  ✂️  Split dataset into train / val / test
Step 3 ──▶  🏋️  Train model and save best checkpoint to models/
Step 4 ──▶  ▶️  Start FastAPI backend (loads model at startup)
Step 5 ──▶  📤  Upload leaf image from frontend (or call API directly)
Step 6 ──▶  📬  Receive prediction, confidence, guidance, explanation & metadata
Step 7 ──▶  📝  Persist prediction log to logs/predictions.csv
```

---

## ⚙️ Setup and Run

### Step 1 — Create a Python 3.10 Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---

### Step 2 — Configure Environment Variables

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
EOF
```

> `utils/llm_explainer.py` reads this `.env` file automatically.

---

### Step 3 — (Optional) Regenerate Scaffold

```bash
python setup_project.py
```

---

### Step 4 — Split Dataset

```bash
python scripts/split_dataset.py
```

---

### Step 5 — Train Model

```bash
python scripts/train_model.py
```

---

### Step 6 — Run Backend API

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

### Step 7 — Run Frontend

From another terminal:

```bash
cd frontend
python -m http.server 5500
```

Then open **[http://127.0.0.1:5500](http://127.0.0.1:5500)** in your browser. 🌐

---

## 🔧 Troubleshooting

<details>
<summary>❌ <code>bad interpreter: .../venv/bin/python3.10: No such file or directory</code></summary>

**Cause:** Virtual environment was created in a different path and then moved.  
**Fix:** Delete that environment and recreate it in the current project path.

</details>

<details>
<summary>❌ <code>No matching distribution found for tensorflow</code></summary>

**Cause:** Running `pip install tensorflow` with an unsupported Python version (commonly 3.14).  
**Fix:** Use Python 3.10 and reinstall dependencies inside `.venv`.

</details>

<details>
<summary>❌ <code>ModuleNotFoundError: No module named 'tensorflow'</code></summary>

**Cause:** Backend started with a Python interpreter that does not have TensorFlow installed.  
**Fix:** Activate `.venv` first, then run backend using `python -m uvicorn ...`.

</details>

<details>
<summary>ℹ️ TensorFlow CUDA/CPU startup messages</summary>

Messages like these are expected on CPU-only machines and do not indicate a failure:

- `Could not find cuda drivers on your machine, GPU will not be used`
- `failed call to cuInit: ... UNKNOWN ERROR (303)`
- `model.compile_metrics will be empty`

Inference still works normally on CPU when the API logs `Model loaded successfully`.

</details>

---

## 🌐 Environment Variables

Variables used by `utils/llm_explainer.py` (loaded from project-root `.env`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Optional | — | Enables OpenAI-generated explanations |
| `OPENAI_MODEL` | Optional | `gpt-4o-mini` | OpenAI model to use for explanations |

> 💡 If `OPENAI_API_KEY` is missing or the request fails, a built-in fallback explanation template is used automatically.

---

## 🖥️ Frontend Notes

`frontend/script.js` currently points to:

```javascript
const API_URL = "http://127.0.0.1:8000/predict"
```

> ⚠️ If your backend runs on a different host or port, update this constant accordingly.

---

## 📝 Notes

- The repository currently includes training, splitting, and single-image prediction scripts.
- Dataset validation and evaluation-report scripts can be added later if needed.

---

<div align="center">

Made for farmers &nbsp;·&nbsp; Powered by 🤖 AI &nbsp;·&nbsp; Built with 🐍 Python

**BeejVaidya** — *Bringing intelligent crop diagnostics to every field* 🌾

</div>
