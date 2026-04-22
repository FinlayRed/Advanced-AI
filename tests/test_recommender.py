"""Smoke tests for the recommendation subsystem.

These are intentionally fast (under a second) so they can run as part of CI
on every push. They are not a substitute for the systematic evaluation in
``src.evaluation`` - they just check the plumbing is intact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.baseline import FrequencyRecencyRecommender
from src.data_generator import GeneratorConfig, generate
from src.evaluation import chronological_split, evaluate
from src.fairness import rerank_for_producer_diversity
from src.mf_model import NMFRecommender
from src.recommender import RecommendationService


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    """Generate a small dataset once per test module run."""
    out = tmp_path_factory.mktemp("data")
    config = GeneratorConfig(n_customers=80, avg_orders_per_customer=12.0, seed=7)
    return generate(out, config=config)


def test_data_generator_produces_expected_columns(small_dataset):
    orders = small_dataset["orders"]
    assert {"order_id", "customer_id", "product_id", "quantity_kg",
            "order_timestamp"}.issubset(orders.columns)
    assert len(orders) > 0


def test_baseline_returns_k_items(small_dataset):
    train, _ = chronological_split(small_dataset["orders"])
    rec = FrequencyRecencyRecommender().fit(train)
    customer = train["customer_id"].iloc[0]
    assert len(rec.recommend_next_order(customer, k=5)) == 5
    assert len(rec.quick_reorder(customer, k=3)) <= 3


def test_baseline_cold_start_does_not_crash(small_dataset):
    train, _ = chronological_split(small_dataset["orders"])
    rec = FrequencyRecencyRecommender().fit(train)
    out = rec.recommend_next_order("UNSEEN_USER", k=5)
    assert len(out) == 5


def test_nmf_beats_random_chance(small_dataset):
    """NMF should at least clear hit_rate > 0.5 on this synthetic data."""
    train, test = chronological_split(small_dataset["orders"])
    rec = NMFRecommender(n_components=8).fit(train)
    metrics = evaluate(rec, train, test, k_values=(10,))
    hit_rate = metrics.loc[
        (metrics["k"] == 10) & (metrics["metric"] == "hit_rate"), "score"
    ].iloc[0]
    assert hit_rate > 0.5


def test_fairness_reranker_increases_producer_diversity(small_dataset):
    products = small_dataset["products"]
    # Construct a degenerate ranking that picks one producer's items first.
    pr0_items = products[products["producer_id"] == products["producer_id"].iloc[0]]
    other_items = products[products["producer_id"] != products["producer_id"].iloc[0]]
    ranking = pr0_items["product_id"].tolist() + other_items["product_id"].tolist()
    reranked = rerank_for_producer_diversity(ranking, products, penalty=0.5, k=5)
    producers_in_top5 = {
        products.loc[products["product_id"] == p, "producer_id"].iloc[0]
        for p in reranked
    }
    # Strict round-robin would give 5 distinct producers; we just want >= 2.
    assert len(producers_in_top5) >= 2


def test_service_round_trip(small_dataset, tmp_path):
    train, _ = chronological_split(small_dataset["orders"])
    service = RecommendationService(
        products=small_dataset["products"],
        feedback_log_path=str(tmp_path / "feedback.jsonl"),
    ).fit(train)
    artefact = tmp_path / "svc.joblib"
    service.save(artefact)
    loaded = RecommendationService.load(artefact)
    customer = train["customer_id"].iloc[0]
    assert loaded.recommend_next_order(customer, k=5) == service.recommend_next_order(
        customer, k=5, log=False
    )
