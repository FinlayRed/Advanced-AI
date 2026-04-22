"""Evaluation harness for the Bristol Food Network recommenders.

Recommendation systems are evaluated chronologically, never with a random
train/test split: the model must predict the future from the past. We train
on the earliest fraction of the order log and evaluate on what each customer
went on to buy in the held-out tail.

Implements four standard top-K metrics:

* Precision@K - fraction of recommended items the user actually bought.
* Recall@K    - fraction of the user's actual purchases we captured in K.
* Hit Rate@K  - did at least one recommendation land?
* NDCG@K      - position-weighted; rewards putting hits near the top.

All metrics are computed per-user then averaged, which is the standard
treatment in the literature (means a heavy buyer doesn't dominate the score).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Recommender(Protocol):
    """Minimal interface every recommender must satisfy to be evaluated."""

    def fit(self, orders: pd.DataFrame) -> "Recommender": ...
    def recommend_next_order(self, customer_id: str, k: int) -> list[str]: ...


def chronological_split(
    orders: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split orders by timestamp into train (earlier) and test (later) sets.

    The split is global, not per-user: the model is evaluated on its ability
    to predict orders placed after a particular wall-clock cutoff. This is
    more realistic than per-user splits because it mirrors how the model
    would be retrained and deployed in production.
    """
    orders = orders.copy()
    orders["order_timestamp"] = pd.to_datetime(orders["order_timestamp"])
    orders = orders.sort_values("order_timestamp")
    cutoff_idx = int(len(orders) * (1 - test_fraction))
    cutoff_time = orders.iloc[cutoff_idx]["order_timestamp"]
    train = orders[orders["order_timestamp"] < cutoff_time].reset_index(drop=True)
    test = orders[orders["order_timestamp"] >= cutoff_time].reset_index(drop=True)
    return train, test


def _ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain at K with binary relevance."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg


def evaluate(
    recommender: Recommender,
    train_orders: pd.DataFrame,
    test_orders: pd.DataFrame,
    k_values: tuple[int, ...] = (5, 10),
    min_train_orders: int = 2,
) -> pd.DataFrame:
    """Evaluate a fitted recommender on a held-out test set.

    Returns a long-form DataFrame with one row per (metric, k) pair so it can
    be easily plotted or pivoted. Customers with fewer than ``min_train_orders``
    in the training period are skipped to keep the evaluation fair - we want
    to measure the recommender, not the cold-start fallback (those should be
    reported separately).
    """
    train_counts = train_orders.groupby("customer_id").size()
    eligible_customers = train_counts[train_counts >= min_train_orders].index

    test_per_customer = (
        test_orders.groupby("customer_id")["product_id"].agg(set).to_dict()
    )

    results: dict[int, dict[str, list[float]]] = {
        k: {"precision": [], "recall": [], "hit_rate": [], "ndcg": []}
        for k in k_values
    }
    n_evaluated = 0

    for customer_id in eligible_customers:
        actual = test_per_customer.get(customer_id)
        if not actual:
            # Customer placed no orders in the test window - skip rather than
            # punish the model for predicting purchases that never happened.
            continue
        n_evaluated += 1
        for k in k_values:
            recs = recommender.recommend_next_order(customer_id, k=k)
            hits = set(recs) & actual
            results[k]["precision"].append(len(hits) / k)
            results[k]["recall"].append(len(hits) / len(actual))
            results[k]["hit_rate"].append(1.0 if hits else 0.0)
            results[k]["ndcg"].append(_ndcg_at_k(recs, actual, k))

    rows = []
    for k, metric_dict in results.items():
        for metric, values in metric_dict.items():
            rows.append({
                "k": k,
                "metric": metric,
                "score": float(np.mean(values)) if values else 0.0,
                "n_customers": n_evaluated,
            })
    return pd.DataFrame(rows)


def producer_concentration(
    recommender: Recommender,
    customer_ids: list[str],
    products: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Measure how concentrated recommendations are across producers.

    A fair recommender should not consistently route customers to the same
    one or two producers. Returns the share of recommendation slots each
    producer receives across the given customer cohort, plus the top-1
    concentration as a single fairness indicator.
    """
    product_to_producer = dict(zip(products["product_id"], products["producer_id"]))
    producer_counts: dict[str, int] = {}
    total_slots = 0
    for customer_id in customer_ids:
        recs = recommender.recommend_next_order(customer_id, k=k)
        for product_id in recs:
            producer = product_to_producer.get(product_id)
            if producer is None:
                continue
            producer_counts[producer] = producer_counts.get(producer, 0) + 1
            total_slots += 1
    if total_slots == 0:
        return pd.DataFrame(columns=["producer_id", "share"])
    rows = [
        {"producer_id": pr, "share": cnt / total_slots}
        for pr, cnt in producer_counts.items()
    ]
    return pd.DataFrame(rows).sort_values("share", ascending=False).reset_index(drop=True)
