from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, TypedDict

from .grading import Condition, assign_grade, grade_from_condition


ActionType = Literal[
    "sell_normal",
    "discount_10_20",
    "clearance_or_remove",
    "manual_review",
]


class InspectionRecord(TypedDict):
    producer_id: str
    product_type: str
    quantity: int
    color_score: float
    size_score: float
    ripeness_score: float
    condition: str
    grade: str
    model_confidence: float
    timestamp: str
    action: ActionType


@dataclass
class ProducerInventory:
    """Per-producer inventory buckets with action recommendations."""

    inventory: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    inspections: List[InspectionRecord] = field(default_factory=list)

    def process_inspection(
        self,
        producer_id: str,
        product_type: str,
        quantity: int,
        color_score: float,
        size_score: float,
        ripeness_score: float,
        model_confidence: float,
        surplus_threshold: int = 100,
        condition: Condition | None = None,
        grade_override: str | None = None,
    ) -> InspectionRecord:
        """Ingest one inspection event and update stock."""
        grade = grade_override or (
            grade_from_condition(condition)
            if condition is not None
            else assign_grade(color_score, size_score, ripeness_score)
        )

        self.inventory.setdefault(producer_id, {})
        self.inventory[producer_id].setdefault(product_type, {"A": 0, "B": 0, "C": 0})
        self.inventory[producer_id][product_type][grade] += quantity

        action = self._recommend_action(
            producer_id=producer_id,
            product_type=product_type,
            grade=grade,
            model_confidence=model_confidence,
            surplus_threshold=surplus_threshold,
        )

        record: InspectionRecord = {
            "producer_id": producer_id,
            "product_type": product_type,
            "quantity": quantity,
            "color_score": float(color_score),
            "size_score": float(size_score),
            "ripeness_score": float(ripeness_score),
            "condition": condition or ("healthy" if grade in {"A", "B"} else "rotten"),
            "grade": grade,
            "model_confidence": float(model_confidence),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
        }
        self.inspections.append(record)
        return record

    def _recommend_action(
        self,
        producer_id: str,
        product_type: str,
        grade: str,
        model_confidence: float,
        surplus_threshold: int,
    ) -> ActionType:
        if model_confidence < 0.60:
            return "manual_review"

        grade_stock = self.inventory[producer_id][product_type][grade]

        if grade == "A":
            return "sell_normal"

        if grade == "B":
            if grade_stock >= surplus_threshold:
                return "discount_10_20"
            return "sell_normal"

        return "clearance_or_remove"

    def get_stock(self, producer_id: str) -> Dict[str, Dict[str, int]]:
        return self.inventory.get(producer_id, {})

    def get_total_stock(self, producer_id: str, product_type: str) -> int:
        buckets = self.inventory.get(producer_id, {}).get(
            product_type, {"A": 0, "B": 0, "C": 0}
        )
        return buckets["A"] + buckets["B"] + buckets["C"]

