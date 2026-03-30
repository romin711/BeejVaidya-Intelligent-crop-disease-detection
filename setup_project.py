from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DIRECTORIES = [
    "dataset/raw",
    "dataset/processed",
    "dataset/split",
    "scripts",
    "models",
    "api",
    "utils",
    "results/logs",
    "results/reports",
]

PYTHON_FILES = [
    "scripts/split_dataset.py",
    "scripts/train_model.py",
    "scripts/predict.py",
    "api/main.py",
    "utils/treatment_map.py",
]

ROOT_FILES = {
    "requirements.txt": "tensorflow\nnumpy\npillow\nscikit-learn\nmatplotlib\nfastapi\nuvicorn\nkaggle\n",
    "README.md": "# BeejVaidya\n\nIntelligent crop disease detection for farmers.\n\n## Project Structure\n\n- `dataset/raw/`: Original dataset files\n- `dataset/processed/`: Cleaned/preprocessed dataset\n- `dataset/split/`: Train/validation/test splits\n- `scripts/`: Dataset, training, evaluation, and prediction scripts\n- `models/`: Saved trained model files\n- `api/`: FastAPI app entrypoint\n- `utils/`: Shared preprocessing and treatment mapping utilities\n- `results/logs/`: Training/inference logs\n- `results/reports/`: Metrics reports and visualizations\n\n## Setup\n\nRun:\n\n```bash\npython setup_project.py\n```\n\nThis creates the complete folder structure and placeholder Python modules.\n",
}


def create_directories() -> None:
    for directory in DIRECTORIES:
        (PROJECT_ROOT / directory).mkdir(parents=True, exist_ok=True)


def create_python_files() -> None:
    for file_path in PYTHON_FILES:
        target = PROJECT_ROOT / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)


def create_root_files() -> None:
    for file_name, content in ROOT_FILES.items():
        target = PROJECT_ROOT / file_name
        if not target.exists():
            target.write_text(content, encoding="utf-8")


def main() -> None:
    create_directories()
    create_python_files()
    create_root_files()
    print(f"Project scaffold created at: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
