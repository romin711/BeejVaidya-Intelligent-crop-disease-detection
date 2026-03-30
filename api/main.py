from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

try:
    from tensorflow.keras.models import load_model
except ImportError:
    # TensorFlow not available - create a fallback
    def load_model(path):
        raise RuntimeError("TensorFlow not installed. Please install with: pip install tensorflow")

from utils.llm_explainer import generate_llm_explanation
from utils.treatment_map import treatment_map


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CLASS_NAMES = ["early_blight", "healthy", "late_blight"]
IMAGE_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.75
LEAF_GREEN_THRESHOLD = 0.05

MODEL_VERSION = "v1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "crop_disease_model.h5"

LOGS_DIR = PROJECT_ROOT / "logs"
PREDICTIONS_LOG_PATH = LOGS_DIR / "predictions.csv"


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(title="Crop Disease Prediction API")


# CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None


# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    image.thumbnail((512, 512))
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def green_pixel_ratio(image: Image.Image) -> float:

    hsv = np.array(image.convert("HSV"))

    h = hsv[..., 0].astype(np.float32) * (179 / 255)
    s = hsv[..., 1]
    v = hsv[..., 2]

    green_mask = (
        (h >= 25) &
        (h <= 90) &
        (s >= 40) &
        (v >= 40)
    )

    return float(np.count_nonzero(green_mask) / green_mask.size)


def get_confidence_level(confidence: float) -> str:

    if confidence >= 0.9:
        return "high"

    if confidence >= 0.75:
        return "medium"

    return "low"


def build_metadata(start_time: float) -> dict:

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "processing_time_ms": int((perf_counter() - start_time) * 1000)
    }


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

def ensure_log_file():

    LOGS_DIR.mkdir(exist_ok=True)

    if not PREDICTIONS_LOG_PATH.exists():

        with open(PREDICTIONS_LOG_PATH, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "image_name",
                "prediction",
                "confidence",
                "confidence_level"
            ])


def append_log(timestamp, image_name, prediction, confidence, level):

    with open(PREDICTIONS_LOG_PATH, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            timestamp,
            image_name,
            prediction,
            "" if confidence is None else round(confidence, 4),
            level
        ])


# --------------------------------------------------
# STARTUP
# --------------------------------------------------

@app.on_event("startup")
def load_prediction_model():

    global model

    ensure_log_file()

    if not MODEL_PATH.exists():
        print(f"⚠ Model not found: {MODEL_PATH}")
        return

    try:
        model = load_model(MODEL_PATH)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load model - {type(e).__name__}: {e}")
        model = None


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok" if model else "error",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(500, "Model not loaded")

    start_time = perf_counter()

    image_name = file.filename or "uploaded_image"
    suffix = Path(image_name).suffix or ".jpg"

    temp_path = None

    try:

        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:

            tmp.write(await file.read())
            temp_path = Path(tmp.name)

        image = Image.open(temp_path).convert("RGB")

    except UnidentifiedImageError:

        raise HTTPException(400, "Invalid image file")

    finally:

        await file.close()

    try:

        # ----------------------------
        # LEAF DETECTION
        # ----------------------------

        if green_pixel_ratio(image) < LEAF_GREEN_THRESHOLD:

            metadata = build_metadata(start_time)

            append_log(
                metadata["timestamp"],
                image_name,
                "UNCERTAIN",
                None,
                "low"
            )

            return {
                "prediction": "UNCERTAIN",
                "message": "Image does not appear to contain a plant leaf",
                **metadata
            }

        # ----------------------------
        # MODEL PREDICTION
        # ----------------------------

        input_tensor = preprocess_image(image)

        preds = model.predict(input_tensor, verbose=0)[0]

        idx = int(np.argmax(preds))

        disease = CLASS_NAMES[idx]

        confidence = float(preds[idx])

        level = get_confidence_level(confidence)

        if confidence < CONFIDENCE_THRESHOLD:

            metadata = build_metadata(start_time)

            append_log(
                metadata["timestamp"],
                image_name,
                "UNCERTAIN",
                confidence,
                level
            )

            return {
                "prediction": "UNCERTAIN",
                "confidence": round(confidence, 4),
                "confidence_level": level,
                "message": "Image unclear or disease not recognized",
                **metadata
            }

        # ----------------------------
        # GUIDANCE
        # ----------------------------

        guidance = treatment_map.get(disease, {})

        treatment = guidance.get(
            "treatment",
            "No treatment guidance available"
        )

        prevention = guidance.get(
            "prevention",
            "No prevention guidance available"
        )

        explanation = generate_llm_explanation(
            disease,
            confidence,
            treatment,
            prevention
        )

        metadata = build_metadata(start_time)

        append_log(
            metadata["timestamp"],
            image_name,
            disease,
            confidence,
            level
        )

        return {
            "prediction": disease,
            "confidence": round(min(confidence, 0.99), 4),
            "confidence_level": level,
            "treatment": treatment,
            "prevention": prevention,
            "explanation": explanation,
            **metadata
        }

    finally:

        if temp_path and temp_path.exists():
            temp_path.unlink()