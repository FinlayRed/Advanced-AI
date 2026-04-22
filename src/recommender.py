"""Public service interface for the Task 1 recommendation subsystem.

This module is the single import the wider Bristol Food Network platform
needs to talk to the recommender. It wraps a fitted model with the producer
fairness re-ranker, the cold-start fallback, and the feedback logger, and
exposes two stable methods:

* ``quick_reorder(customer_id, k)`` -> list[str]
* ``recommend_next_order(customer_id, k)`` -> list[str]

Both return product_ids, in ranked order best first. The whole service is
serialisable with joblib so AI engineers (per the case study brief) can train
a new model offline and drop the resulting .joblib file into the deployed
service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from .baseline import FrequencyRecencyRecommender
from .fairness import rerank_for_producer_diversity
from .feedback import FeedbackLogger
from .mf_model import NMFRecommender


ModelName = Literal["baseline", "nmf"]


@dataclass
class RecommendationService:
    """Production-facing wrapper around the underlying recommender models.

    Holds both models so the service can A/B test or switch at runtime
    without re-loading. The active model is configurable via ``active_model``.
    """

    products: pd.DataFrame
    baseline: FrequencyRecencyRecommender = field(default_factory=FrequencyRecencyRecommender)
    nmf: NMFRecommender = field(default_factory=NMFRecommender)
    active_model: ModelName = "nmf"
    fairness_penalty: float = 0.6
    feedback_log_path: str = "logs/feedback.jsonl"
    _logger: FeedbackLogger = field(init=False)

    def __post_init__(self) -> None:
        self._logger = FeedbackLogger(self.feedback_log_path)

    def fit(self, orders: pd.DataFrame) -> "RecommendationService":
        """Fit both underlying models on the same order log."""
        all_products = self.products["product_id"].tolist()
        self.baseline.fit(orders, all_product_ids=all_products)
        self.nmf.fit(orders)
        return self

    def _underlying(self):
        return self.baseline if self.active_model == "baseline" else self.nmf

    def quick_reorder(
        self,
        customer_id: str,
        k: int = 5,
        log: bool = True,
    ) -> list[str]:
        """Top-k frequent items for a one-tap re-order experience."""
        recs = self._underlying().quick_reorder(customer_id, k=k)
        if log:
            self._logger.log_recommendation(
                customer_id, recs, model_name=f"{self.active_model}.quick_reorder"
            )
        return recs

    def recommend_next_order(
        self,
        customer_id: str,
        k: int = 10,
        log: bool = True,
        fairness: bool = True,
    ) -> list[str]:
        """Top-k recommendations for the customer's next order.

        Generates 3K candidates from the underlying model, then optionally
        re-ranks for producer diversity and trims to K. The 3K oversample
        is what lets the fairness re-ranker actually move things around -
        without it the re-ranker has nothing to work with.
        """
        candidates = self._underlying().recommend_next_order(
            customer_id, k=min(k * 3, len(self.products))
        )
        if fairness:
            candidates = rerank_for_producer_diversity(
                candidates,
                products=self.products,
                penalty=self.fairness_penalty,
                k=k,
            )
        else:
            candidates = candidates[:k]
        if log:
            self._logger.log_recommendation(
                customer_id, candidates, model_name=f"{self.active_model}.next_order"
            )
        return candidates

    def record_outcome(
        self,
        event_id: str,
        accepted: list[str],
        added_outside_recommendation: list[str],
    ) -> None:
        """Pass-through to the feedback logger - exposed for the UI/API layer."""
        self._logger.log_outcome(event_id, accepted, added_outside_recommendation)

    def save(self, path: str | Path) -> None:
        """Serialise the entire fitted service to a single .joblib file.

        This is the artefact AI engineers upload to deploy a new model
        version per the case study's Task 3 requirement.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "RecommendationService":
        """Load a previously saved service."""
        return joblib.load(path)
