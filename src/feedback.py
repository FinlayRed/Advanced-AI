"""Override and feedback logging.

The case study asks: 'If a user overrides the model's prediction, what are
you going to do about that? How does that relate to monitoring performance
over time?'

This module is the data plumbing for that question. Every time the UI shows
recommendations to a user we record (a) what we showed and (b) what they
actually did - did they accept the suggestion, swap one item, ignore the
list entirely, or replace it with something completely different. The
resulting log is the training signal for the next retrain and the input to
performance-drift monitoring.

The implementation here is intentionally minimal (newline-delimited JSON
appended to a file) so it can be swapped for a real database table when the
service is integrated into the wider DESD platform without changing the
calling code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FeedbackLogger:
    """Append-only JSONL feedback log."""

    log_path: Path | str = "logs/feedback.jsonl"

    def __post_init__(self) -> None:
        self.log_path = Path(self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_recommendation(
        self,
        customer_id: str,
        recommended: list[str],
        model_name: str,
    ) -> str:
        """Record what the recommender suggested. Returns an event id."""
        event_id = f"rec-{_utc_now_iso()}-{customer_id}"
        record = {
            "event_id": event_id,
            "event_type": "recommendation_shown",
            "timestamp": _utc_now_iso(),
            "customer_id": customer_id,
            "model_name": model_name,
            "recommended": recommended,
        }
        self._append(record)
        return event_id

    def log_outcome(
        self,
        event_id: str,
        accepted: list[str],
        added_outside_recommendation: list[str],
    ) -> None:
        """Record what the customer ended up putting in their basket.

        ``accepted`` is the subset of the recommendations the customer kept;
        ``added_outside_recommendation`` is anything they added that was not
        in the recommended list. The pair tells us both the precision of the
        recommendation event and what alternatives the customer preferred -
        gold-standard training data for the next retrain.
        """
        record = {
            "event_id": event_id,
            "event_type": "outcome_recorded",
            "timestamp": _utc_now_iso(),
            "accepted": accepted,
            "added_outside_recommendation": added_outside_recommendation,
        }
        self._append(record)

    def _append(self, record: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
