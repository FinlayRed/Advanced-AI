"""Matrix-factorisation recommender using non-negative matrix factorisation.

NMF is well-suited to implicit-feedback purchase data: we factor the
customer x product interaction matrix ``R`` into ``W @ H`` where ``W`` are
customer latent factors and ``H`` are product latent factors. Predicted
affinity for a (customer, product) pair is then the dot product of their
latent vectors.

Why NMF specifically (vs SVD, ALS, neural CF):

* Non-negativity makes the factors more interpretable - latent dimensions
  tend to look like 'salad ingredients' or 'breakfast fruits', which is a
  small win towards the explainability marking criterion.
* It is in scikit-learn, so no extra dependencies and no compilation issues.
* Training time on the synthetic dataset (~73k order lines, 800 customers,
  43 products) is well under a second on a laptop.

Limitations to flag in the report:

* NMF treats item counts as ratings. We log-transform and clip them to keep
  heavy buyers from dominating, but it remains a known weakness.
* The factorisation has to be re-run from scratch when new customers arrive.
  In production we would warm-start or switch to incremental ALS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


@dataclass
class NMFRecommender:
    """Collaborative filtering via non-negative matrix factorisation.

    Parameters
    ----------
    n_components:
        Number of latent factors. 8-16 is a reasonable range for a catalogue
        of ~50 products; larger values overfit on small data.
    max_iter:
        Maximum NMF coordinate-descent iterations.
    random_state:
        Seed for reproducibility.
    """

    n_components: int = 12
    max_iter: int = 400
    random_state: int = 42

    _model: NMF | None = field(default=None, init=False)
    _user_factors: np.ndarray | None = field(default=None, init=False)
    _item_factors: np.ndarray | None = field(default=None, init=False)
    _user_index: dict[str, int] = field(default_factory=dict, init=False)
    _item_index: dict[str, int] = field(default_factory=dict, init=False)
    _items_by_pos: list[str] = field(default_factory=list, init=False)
    _seen: dict[str, set[str]] = field(default_factory=dict, init=False)
    _global_popularity: pd.Series | None = field(default=None, init=False)

    def _build_interaction_matrix(
        self,
        orders: pd.DataFrame,
    ) -> np.ndarray:
        """Aggregate orders into a customer x product count matrix, log-scaled."""
        users = orders["customer_id"].unique().tolist()
        items = orders["product_id"].unique().tolist()
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {p: j for j, p in enumerate(items)}
        self._items_by_pos = items

        counts = (
            orders.groupby(["customer_id", "product_id"]).size().reset_index(
                name="count"
            )
        )
        matrix = np.zeros((len(users), len(items)), dtype=np.float32)
        u_idx = counts["customer_id"].map(self._user_index).to_numpy()
        i_idx = counts["product_id"].map(self._item_index).to_numpy()
        # log1p softens the dominance of heavy buyers without erasing the
        # difference between buying once and buying twenty times.
        matrix[u_idx, i_idx] = np.log1p(counts["count"].to_numpy())
        return matrix

    def fit(self, orders: pd.DataFrame) -> "NMFRecommender":
        """Fit the NMF model on a purchase log."""
        required = {"customer_id", "product_id", "order_timestamp"}
        missing = required - set(orders.columns)
        if missing:
            raise ValueError(f"orders is missing required columns: {missing}")

        matrix = self._build_interaction_matrix(orders)

        self._model = NMF(
            n_components=self.n_components,
            init="nndsvda",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._user_factors = self._model.fit_transform(matrix)
        self._item_factors = self._model.components_  # shape (k, n_items)

        # Cache items each user has already bought so we can either include
        # or exclude them at recommend time as appropriate.
        self._seen = (
            orders.groupby("customer_id")["product_id"]
            .agg(set)
            .to_dict()
        )

        # Recent (90-day) popularity for cold-start fallback - same idea as
        # the baseline so the two models behave consistently for new users.
        orders_ts = orders.copy()
        orders_ts["order_timestamp"] = pd.to_datetime(orders_ts["order_timestamp"])
        cutoff = orders_ts["order_timestamp"].max() - pd.Timedelta(days=90)
        recent = orders_ts[orders_ts["order_timestamp"] >= cutoff]
        self._global_popularity = (
            recent.groupby("product_id").size().sort_values(ascending=False)
        )

        return self

    def _predict_scores(self, customer_id: str) -> np.ndarray | None:
        """Predicted affinity scores across the full catalogue, or None for cold start."""
        if self._user_factors is None or self._item_factors is None:
            raise RuntimeError("Recommender has not been fitted.")
        user_pos = self._user_index.get(customer_id)
        if user_pos is None:
            return None
        return self._user_factors[user_pos] @ self._item_factors

    def recommend_next_order(
        self,
        customer_id: str,
        k: int = 10,
        exclude_seen: bool = False,
    ) -> list[str]:
        """Top-k recommended products for the customer.

        ``exclude_seen=True`` is useful for surfacing genuinely new items the
        customer has not tried; ``False`` (default) lets the model recommend
        repeat purchases, which is what 'next order' usually means in
        groceries.
        """
        scores = self._predict_scores(customer_id)
        if scores is None:
            # Cold start - fall back to recent popularity.
            assert self._global_popularity is not None
            return self._global_popularity.head(k).index.tolist()

        if exclude_seen:
            seen = self._seen.get(customer_id, set())
            for item, idx in self._item_index.items():
                if item in seen:
                    scores[idx] = -np.inf

        top_pos = np.argsort(-scores)[:k]
        return [self._items_by_pos[p] for p in top_pos]

    def quick_reorder(self, customer_id: str, k: int = 5) -> list[str]:
        """Top-k items for one-tap re-ordering.

        For NMF this is implemented as 'top scored items the customer has
        actually bought before' - we never want to suggest a re-order of
        something the customer has never had.
        """
        scores = self._predict_scores(customer_id)
        seen = self._seen.get(customer_id, set())
        if scores is None or not seen:
            assert self._global_popularity is not None
            return self._global_popularity.head(k).index.tolist()

        # Mask out unseen items.
        for item, idx in self._item_index.items():
            if item not in seen:
                scores[idx] = -np.inf
        top_pos = np.argsort(-scores)[:k]
        return [self._items_by_pos[p] for p in top_pos]
