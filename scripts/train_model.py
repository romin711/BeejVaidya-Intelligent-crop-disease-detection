from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 15
CLASS_NAMES = ["early_blight", "healthy", "late_blight"]


class EpochMetricsLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0.0)
        train_acc = logs.get("accuracy", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Training accuracy: {train_acc:.4f} - "
            f"Validation accuracy: {val_acc:.4f} - "
            f"Loss: {loss:.4f}"
        )


def build_model() -> Model:
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        weights="imagenet",
        include_top=False,
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(3, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_generators(split_dir: Path):
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    test_dir = split_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise SystemExit(
            "Missing dataset split folders. Expected: "
            "dataset/split/train, dataset/split/val, dataset/split/test"
        )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    split_dir = project_root / "dataset" / "split"
    model_output = project_root / "models" / "crop_disease_model.h5"
    model_output.parent.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, _ = build_generators(split_dir)
    model = build_model()

    checkpoint = ModelCheckpoint(
        filepath=model_output,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, EpochMetricsLogger()],
        verbose=0,
    )

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
