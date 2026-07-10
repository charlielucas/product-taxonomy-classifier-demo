"""Prediction validation and review queue helpers."""

from __future__ import annotations


ALLOWED_CATEGORIES = {
    "Brake System",
    "Cabin Comfort",
    "Drivetrain",
    "Electrical",
    "Engine Cooling",
    "Exterior",
    "Fluids",
    "Tools",
}

ALLOWED_MANUFACTURERS = {
    "Apex AutoWorks",
    "BluePeak",
    "Harbor Works",
    "Northstar Components",
    "Orbit Parts",
    "TerraLine",
}


def review_reasons(
    category: str,
    category_confidence: float,
    manufacturer: str,
    manufacturer_confidence: float,
    threshold: float = 0.72,
) -> list[str]:
    reasons: list[str] = []
    if category not in ALLOWED_CATEGORIES:
        reasons.append("category outside allowed taxonomy")
    if manufacturer not in ALLOWED_MANUFACTURERS:
        reasons.append("manufacturer outside allowed list")
    if category_confidence < threshold:
        reasons.append("low category confidence")
    if manufacturer_confidence < threshold:
        reasons.append("low manufacturer confidence")
    return reasons


def needs_review(reasons: list[str]) -> bool:
    return bool(reasons)
