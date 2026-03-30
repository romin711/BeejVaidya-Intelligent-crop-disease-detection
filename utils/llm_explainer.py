from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error, request


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()


def _build_prompt(disease: str, confidence: float, treatment: str, prevention: str) -> str:
    return (
        "You are an agricultural assistant helping farmers understand plant diseases.\n\n"
        f"Disease detected: {disease}\n"
        f"Model confidence: {confidence:.4f}\n\n"
        "Explain the disease in simple language and include:\n\n"
        "1. What the disease is\n"
        "2. Why it happens\n"
        "3. What the farmer should do immediately\n"
        "4. Prevention tips\n\n"
        "Keep the explanation short and easy to understand.\n\n"
        f"Treatment guidance: {treatment}\n"
        f"Prevention guidance: {prevention}"
    )


def _fallback_explanation(disease: str, treatment: str, prevention: str) -> str:
    disease_title = disease.replace("_", " ").title()
    return (
        f"Disease detected: {disease_title}.\n\n"
        "This can affect plant health and spread faster in poor weather or field conditions.\n\n"
        "Do this now:\n"
        f"{treatment}\n\n"
        "Prevention tips:\n"
        f"{prevention}"
    )


def generate_llm_explanation(
    disease: str,
    confidence: float,
    treatment: str,
    prevention: str,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _fallback_explanation(disease, treatment, prevention)

    payload = {
        "model": os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        "messages": [
            {
                "role": "system",
                "content": "You are an agricultural assistant helping farmers understand plant diseases.",
            },
            {
                "role": "user",
                "content": _build_prompt(disease, confidence, treatment, prevention),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 220,
    }

    req = request.Request(
        OPENAI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return _fallback_explanation(disease, treatment, prevention)

    choices = data.get("choices", [])
    if not choices:
        return _fallback_explanation(disease, treatment, prevention)

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = " ".join(part.strip() for part in text_parts if part.strip())

    explanation = str(content).strip()
    if not explanation:
        return _fallback_explanation(disease, treatment, prevention)

    return explanation
