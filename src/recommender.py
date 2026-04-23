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
import numpy as np
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

    def explain_quick_reorder(self, customer_id: str, k: int = 5) -> list[dict]:
        """Return quick re-order items with simple explanation payloads."""
        ranked = self.quick_reorder(customer_id, k=k, log=False)
        return [
            self._build_explanation(
                customer_id=customer_id,
                product_id=product_id,
                rank=rank,
                route="quick_reorder",
                fairness_applied=False,
            )
            for rank, product_id in enumerate(ranked, start=1)
        ]

    def explain_next_order(
        self,
        customer_id: str,
        k: int = 10,
        fairness: bool = True,
    ) -> list[dict]:
        """Return next-order recommendations with XAI payloads."""
        raw_candidates = self._underlying().recommend_next_order(
            customer_id, k=min(k * 3, len(self.products))
        )
        if fairness:
            final_ranked = rerank_for_producer_diversity(
                raw_candidates,
                products=self.products,
                penalty=self.fairness_penalty,
                k=k,
            )
        else:
            final_ranked = raw_candidates[:k]

        raw_positions = {product_id: pos for pos, product_id in enumerate(raw_candidates, start=1)}
        return [
            self._build_explanation(
                customer_id=customer_id,
                product_id=product_id,
                rank=rank,
                route="next_order",
                fairness_applied=fairness,
                raw_rank=raw_positions.get(product_id),
            )
            for rank, product_id in enumerate(final_ranked, start=1)
        ]

    def _product_record(self, product_id: str) -> dict:
        match = self.products.loc[self.products["product_id"] == product_id]
        if match.empty:
            return {"product_id": product_id}
        row = match.iloc[0]
        return row.to_dict()

    def _score_for_product(self, customer_id: str, product_id: str, route: str) -> float | None:
        if self.active_model == "baseline":
            if route == "quick_reorder":
                scores = self.baseline._customer_scores(customer_id)
                if product_id in scores.index:
                    return float(scores.loc[product_id])
                return None

            personal = self.baseline._customer_scores(customer_id)
            if self.baseline._global_popularity is None:
                return None
            prior = self.baseline._global_popularity / self.baseline._global_popularity.max()
            personal_norm = personal / personal.max() if not personal.empty else pd.Series(dtype=float)
            combined = personal_norm.add(prior * 0.15, fill_value=0.0)
            if product_id in combined.index:
                return float(combined.loc[product_id])
            return None

        scores = self.nmf._predict_scores(customer_id)
        if scores is None:
            if self.nmf._global_popularity is not None and product_id in self.nmf._global_popularity.index:
                return float(self.nmf._global_popularity.loc[product_id])
            return None
        item_index = self.nmf._item_index.get(product_id)
        if item_index is None:
            return None
        return float(scores[item_index])

    def _reason_text(self, customer_id: str, product_id: str, route: str, fairness_applied: bool, raw_rank: int | None) -> str:
        if self.active_model == "baseline":
            if route == "quick_reorder":
                text = "Recommended because this item appears frequently and recently in the customer's order history."
            else:
                text = "Recommended by combining the customer's repeat-purchase pattern with recent marketplace demand."
        else:
            if customer_id in self.nmf._user_index:
                text = "Recommended because the collaborative filtering model found similar basket patterns across customers."
            else:
                text = "Recommended from recent marketplace popularity because the customer has limited history."
        if fairness_applied and raw_rank is not None and raw_rank > 1:
            text += f" Producer-diversity re-ranking moved it from position {raw_rank} to improve marketplace fairness."
        return text

    def _build_explanation(
        self,
        customer_id: str,
        product_id: str,
        rank: int,
        route: str,
        fairness_applied: bool,
        raw_rank: int | None = None,
    ) -> dict:
        product = self._product_record(product_id)
        score = self._score_for_product(customer_id, product_id, route)
        confidence = None if score is None else round(float(1.0 / (1.0 + np.exp(-score))), 4)
        return {
            "product_id": product_id,
            "rank": rank,
            "category": product.get("category"),
            "producer_id": product.get("producer_id"),
            "price_per_kg": product.get("price_per_kg"),
            "model_name": self.active_model,
            "route": route,
            "raw_rank": raw_rank,
            "score": None if score is None else round(score, 4),
            "confidence": confidence,
            "reason_text": self._reason_text(
                customer_id=customer_id,
                product_id=product_id,
                route=route,
                fairness_applied=fairness_applied,
                raw_rank=raw_rank,
            ),
            "reason_codes": [
                route,
                self.active_model,
                "fairness_reranked" if fairness_applied and raw_rank and raw_rank != rank else "direct_rank",
            ],
        }

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
