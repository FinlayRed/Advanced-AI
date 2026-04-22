"""Frequency-recency baseline recommender.

This is deliberately simple: for each customer, score every product they have
ever bought by ``count * exp(-lambda * days_since_last_purchase)``. The result
is a strong baseline for both 'frequent staples' (high count) and 'recent
favourites' (small days_since_last). It also doubles as the implementation of
the ``quick_reorder`` endpoint required by the case study brief.

Two endpoints are exposed:

* ``quick_reorder(customer_id, k)`` - the customer's top-k recurring items,
  ranked by frequency-recency score. This is what the UI should put behind a
  one-tap re-order button.
* ``recommend_next_order(customer_id, k)`` - same scoring, but combined with a
  popularity prior so the model can fall back gracefully for new or sparse
  customers (cold start).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FrequencyRecencyRecommender:
    """Score products by how often and how recently a customer has bought them.

    Parameters
    ----------
    decay_days:
        Half-life of the recency decay in days. Smaller values weight recent
        purchases more heavily. Default 30 days roughly matches the cadence
        of weekly-to-monthly grocery shopping.
    """

    decay_days: float = 30.0
    _orders: pd.DataFrame | None = field(default=None, init=False)
    _reference_date: pd.Timestamp | None = field(default=None, init=False)
    _global_popularity: pd.Series | None = field(default=None, init=False)
    _all_products: list[str] = field(default_factory=list, init=False)

    def fit(
        self,
        orders: pd.DataFrame,
        all_product_ids: list[str] | None = None,
    ) -> "FrequencyRecencyRecommender":
        """Learn from an order log.

        ``orders`` must contain ``customer_id``, ``product_id`` and
        ``order_timestamp`` columns. ``all_product_ids`` lets you pass the full
        catalogue so cold-start fallbacks can recommend items no one has
        bought yet (otherwise we use only items present in the order log).
        """
        required = {"customer_id", "product_id", "order_timestamp"}
        missing = required - set(orders.columns)
        if missing:
            raise ValueError(f"orders is missing required columns: {missing}")

        self._orders = orders.copy()
        self._orders["order_timestamp"] = pd.to_datetime(
            self._orders["order_timestamp"]
        )
        self._reference_date = self._orders["order_timestamp"].max()

        # Global popularity in the most recent 90 days, used as the cold-start
        # prior. Recent popularity is more useful than all-time popularity in
        # a seasonal market.
        recent_cutoff = self._reference_date - pd.Timedelta(days=90)
        recent = self._orders[self._orders["order_timestamp"] >= recent_cutoff]
        self._global_popularity = (
            recent.groupby("product_id").size().sort_values(ascending=False)
        )

        if all_product_ids is None:
            self._all_products = self._orders["product_id"].unique().tolist()
        else:
            self._all_products = list(all_product_ids)

        return self

    def _customer_scores(self, customer_id: str) -> pd.Series:
        """Compute frequency-recency scores for everything this customer has bought."""
        if self._orders is None or self._reference_date is None:
            raise RuntimeError("Recommender has not been fitted.")

        cust_orders = self._orders[self._orders["customer_id"] == customer_id]
        if cust_orders.empty:
            return pd.Series(dtype=float)

        days_since = (
            self._reference_date - cust_orders["order_timestamp"]
        ).dt.total_seconds() / 86400.0
        # Half-life decay: weight = 0.5 ** (days_since / decay_days)
        weight = np.power(0.5, days_since / self.decay_days)
        cust_orders = cust_orders.assign(_weight=weight.values)
        return cust_orders.groupby("product_id")["_weight"].sum().sort_values(
            ascending=False
        )

    def quick_reorder(self, customer_id: str, k: int = 5) -> list[str]:
        """Top-k items the customer is likely to want to re-order right now."""
        scores = self._customer_scores(customer_id)
        return scores.head(k).index.tolist()

    def recommend_next_order(
        self,
        customer_id: str,
        k: int = 10,
        explore_weight: float = 0.15,
    ) -> list[str]:
        """Top-k recommendations blending personal history with popular items.

        ``explore_weight`` controls the strength of the popularity prior. For
        cold-start customers (no history) the prior is the only signal so we
        return the global top-k. For known customers their history dominates
        and the prior just nudges some unseen items into the long tail.
        """
        if self._global_popularity is None:
            raise RuntimeError("Recommender has not been fitted.")

        personal = self._customer_scores(customer_id)
        if personal.empty:
            # Cold start - return globally popular items.
            return self._global_popularity.head(k).index.tolist()

        # Normalise both signals to [0, 1] before combining so explore_weight
        # behaves intuitively regardless of how many orders the customer has.
        personal_norm = personal / personal.max()
        prior = self._global_popularity / self._global_popularity.max()
        combined = personal_norm.add(prior * explore_weight, fill_value=0.0)
        return combined.sort_values(ascending=False).head(k).index.tolist()
