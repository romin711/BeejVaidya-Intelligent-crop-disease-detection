from __future__ import annotations


CAUSE_MAP = {
    "early_blight": "This disease is caused by a fungal infection that spreads in warm and humid conditions.",
    "late_blight": "This disease spreads quickly in cool, wet weather and can damage leaves and stems fast.",
    "healthy": "Your plant looks healthy and does not show visible signs of major disease.",
}


def generate_explanation(disease: str, treatment: str, prevention: str) -> str:
    disease_title = disease.replace("_", " ").title()
    disease_cause = CAUSE_MAP.get(
        disease, "This condition can spread through unfavorable weather and poor field hygiene."
    )

    return (
        f"Disease detected: {disease_title}.\n\n"
        f"{disease_cause}\n\n"
        "Recommended treatment:\n"
        f"{treatment}\n\n"
        "Prevention tips:\n"
        f"{prevention}"
    )
