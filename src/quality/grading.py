from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


Condition = Literal["healthy", "rotten"]


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds for grade assignment."""

    color_b: float = 75.0
    size_b: float = 80.0
    ripeness_b: float = 70.0

    color_c: float = 65.0
    size_c: float = 70.0
    ripeness_c: float = 60.0


def assign_grade(
    color: float,
    size: float,
    ripeness: float,
    thresholds: QualityThresholds = QualityThresholds(),
) -> str:
    """Assign A/B/C grade from quality scores."""
    if color < thresholds.color_c or size < thresholds.size_c or ripeness < thresholds.ripeness_c:
        return "C"

    if color < thresholds.color_b or size < thresholds.size_b or ripeness < thresholds.ripeness_b:
        return "B"

    return "A"


def condition_from_label(label: str) -> Condition:
    """Normalize model class labels to the binary quality decision."""
    normalized = label.strip().lower()
    if "rotten" in normalized or "rot" in normalized or "spoiled" in normalized:
        return "rotten"
    if "healthy" in normalized or "fresh" in normalized:
        return "healthy"
    return "rotten"


def grade_from_condition(condition: Condition) -> str:
    return "A" if condition == "healthy" else "C"


def quality_breakdown(color: float, size: float, ripeness: float) -> Dict[str, float]:
    return {
        "color_score": round(float(color), 2),
        "size_score": round(float(size), 2),
        "ripeness_score": round(float(ripeness), 2),
    }

