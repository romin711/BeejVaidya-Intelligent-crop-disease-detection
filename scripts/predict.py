from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


CLASS_NAMES = ["early_blight", "healthy", "late_blight"]
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.75
LEAF_GREEN_THRESHOLD = 0.10


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def green_pixel_ratio(image: Image.Image) -> float:
    hsv_image = np.array(image.convert("HSV"), dtype=np.uint8)
    h_channel = hsv_image[..., 0].astype(np.float32) * (179.0 / 255.0)
    s_channel = hsv_image[..., 1]
    v_channel = hsv_image[..., 2]

    green_mask = (
        (h_channel >= 25)
        & (h_channel <= 90)
        & (s_channel >= 40)
        & (s_channel <= 255)
        & (v_channel >= 40)
        & (v_channel <= 255)
    )
    return float(np.count_nonzero(green_mask) / green_mask.size)


def predict_image(model_path: Path, image_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    green_ratio = green_pixel_ratio(image)
    if green_ratio < LEAF_GREEN_THRESHOLD:
        print("Prediction: UNCERTAIN")
        print("Message: Image does not appear to contain a plant leaf.")
        return

    model = load_model(model_path)
    input_batch = preprocess_image(image)
    prediction_vector = model.predict(input_batch, verbose=0)[0]

    if len(prediction_vector) != len(CLASS_NAMES):
        raise SystemExit(
            f"Model output size ({len(prediction_vector)}) does not match "
            f"class count ({len(CLASS_NAMES)})."
        )

    top_index = int(np.argmax(prediction_vector))
    top_class = CLASS_NAMES[top_index]
    top_probability = float(prediction_vector[top_index])
    confidence = top_probability * 100.0

    if top_probability < CONFIDENCE_THRESHOLD:
        print("Prediction: UNCERTAIN")
        print("Message: Image unclear or disease not recognized.")
        return

    print(f"Prediction: {top_class}")
    print(f"Confidence: {confidence:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-image prediction using crop disease model."
    )
    parser.add_argument("image_path", type=Path, help="Path to input image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "crop_disease_model.h5"
    image_path = args.image_path

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")

    predict_image(model_path, image_path)


if __name__ == "__main__":
    main()
