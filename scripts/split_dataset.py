from __future__ import annotations

import random
import shutil
from pathlib import Path


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def split_counts(total: int) -> tuple[int, int, int]:
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "dataset" / "raw"
    split_dir = project_root / "dataset" / "split"

    if not raw_dir.exists():
        raise SystemExit(f"Raw dataset directory not found: {raw_dir}")

    class_dirs = sorted([path for path in raw_dir.iterdir() if path.is_dir()])
    if not class_dirs:
        raise SystemExit(f"No class folders found in: {raw_dir}")

    if split_dir.exists():
        shutil.rmtree(split_dir)

    split_names = ("train", "val", "test")
    summary: dict[str, dict[str, int]] = {split: {} for split in split_names}

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [path for path in class_dir.iterdir() if is_image_file(path)]
        random.shuffle(images)

        train_count, val_count, _ = split_counts(len(images))
        train_images = images[:train_count]
        val_images = images[train_count : train_count + val_count]
        test_images = images[train_count + val_count :]

        groups = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        for split_name, files in groups.items():
            target_dir = split_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for file_path in files:
                shutil.copy2(file_path, target_dir / file_path.name)
            summary[split_name][class_name] = len(files)

    print("Dataset Split Summary")
    print("---------------------")
    print()

    print("TRAIN")
    for class_name in sorted(summary["train"]):
        print(f"{class_name} : {summary['train'][class_name]}")
    print()

    print("VALIDATION")
    for class_name in sorted(summary["val"]):
        print(f"{class_name} : {summary['val'][class_name]}")
    print()

    print("TEST")
    for class_name in sorted(summary["test"]):
        print(f"{class_name} : {summary['test'][class_name]}")


if __name__ == "__main__":
    main()
