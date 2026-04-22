"""End-to-end demo: generate data, train both models, evaluate, compare.

Run from the project root:

    python -m src.demo

This script is the single entry point for the in-class demo of Task 1. It:

1. Generates the synthetic purchase dataset.
2. Performs a chronological train/test split.
3. Fits the frequency-recency baseline AND the NMF model on the training set.
4. Evaluates both on the test set with Precision@K, Recall@K, Hit Rate@K, NDCG@K.
5. Demonstrates producer-fairness re-ranking and prints the diversity impact.
6. Demonstrates cold-start fallback for an unseen customer.
7. Demonstrates the override-logging flow.
8. Saves the fitted service to a .joblib artefact (the file Finlay's
   integration layer will load).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data_generator import generate
from src.baseline import FrequencyRecencyRecommender
from src.evaluation import (
    chronological_split,
    evaluate,
    producer_concentration,
)
from src.fairness import rerank_for_producer_diversity
from src.mf_model import NMFRecommender
from src.recommender import RecommendationService


def section(title: str) -> None:
    """Pretty printing for demo output."""
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    section("Step 1: Generating synthetic purchase history")
    data = generate("data")
    customers, products, orders = data["customers"], data["products"], data["orders"]
    print(f"  Customers: {len(customers):,}")
    print(f"  Products:  {len(products):,} ({products['category'].nunique()} categories)")
    print(f"  Producers: {products['producer_id'].nunique()}")
    print(f"  Orders:    {len(orders):,} order lines")

    section("Step 2: Chronological train/test split (last 20% as test)")
    train, test = chronological_split(orders, test_fraction=0.2)
    print(f"  Train:     {len(train):,} order lines  ({train['order_timestamp'].min()} -> {train['order_timestamp'].max()})")
    print(f"  Test:      {len(test):,} order lines  ({test['order_timestamp'].min()} -> {test['order_timestamp'].max()})")

    section("Step 3: Fitting both models on training data")
    baseline = FrequencyRecencyRecommender(decay_days=30.0).fit(train)
    nmf = NMFRecommender(n_components=12).fit(train)
    print("  Baseline:  FrequencyRecencyRecommender (half-life 30 days)")
    print(f"  NMF:       latent dims = {nmf.n_components}, max_iter = {nmf.max_iter}")

    section("Step 4: Evaluation on held-out test set")
    baseline_metrics = evaluate(baseline, train, test, k_values=(5, 10))
    nmf_metrics = evaluate(nmf, train, test, k_values=(5, 10))

    comparison = baseline_metrics.merge(
        nmf_metrics,
        on=["k", "metric"],
        suffixes=("_baseline", "_nmf"),
    )
    comparison = comparison[[
        "k", "metric", "score_baseline", "score_nmf", "n_customers_baseline"
    ]].rename(columns={"n_customers_baseline": "n_customers"})
    comparison["score_baseline"] = comparison["score_baseline"].round(4)
    comparison["score_nmf"] = comparison["score_nmf"].round(4)
    print(comparison.to_string(index=False))

    section("Step 5: Producer-fairness re-ranking")
    sample_customer = train["customer_id"].value_counts().index[0]
    raw_top10 = nmf.recommend_next_order(sample_customer, k=10)
    candidates = nmf.recommend_next_order(sample_customer, k=30)
    reranked = rerank_for_producer_diversity(
        candidates, products=products, penalty=0.6, k=10
    )

    p_to_pr = dict(zip(products["product_id"], products["producer_id"]))
    print(f"  Customer: {sample_customer}")
    print(f"  Without fairness, producers in top-10: "
          f"{sorted({p_to_pr[p] for p in raw_top10})}")
    print(f"  With fairness, producers in top-10:    "
          f"{sorted({p_to_pr[p] for p in reranked})}")

    cohort = train["customer_id"].value_counts().head(100).index.tolist()
    conc_no = producer_concentration(nmf, cohort, products, k=10)
    print(f"\n  Top producer share across 100-customer cohort:")
    print(f"    Without fairness: {conc_no['share'].iloc[0] * 100:.1f}% "
          f"({conc_no['producer_id'].iloc[0]})")

    section("Step 6: Cold-start fallback")
    new_customer = "C9999_BRAND_NEW"
    cold_recs = baseline.recommend_next_order(new_customer, k=5)
    print(f"  Brand-new customer {new_customer} recommendations:")
    for i, pid in enumerate(cold_recs, 1):
        cat = products.loc[products["product_id"] == pid, "category"].iloc[0]
        print(f"    {i}. {pid} ({cat})")

    section("Step 7: Override / feedback logging")
    service = RecommendationService(products=products).fit(train)
    recs = service.recommend_next_order(sample_customer, k=5)
    print(f"  Recommended to {sample_customer}: {recs}")
    # Simulate the customer accepting two and adding something else.
    service.record_outcome(
        event_id="rec-demo-event",
        accepted=recs[:2],
        added_outside_recommendation=["P0007"],
    )
    log_path = Path("logs/feedback.jsonl")
    print(f"  Wrote {log_path.stat().st_size} bytes to {log_path}")
    print("  (This is the data we would use to retrain and to monitor drift.)")

    section("Step 8: Saving the deployable service artefact")
    artefact = Path("data/recommendation_service.joblib")
    service.save(artefact)
    print(f"  Saved -> {artefact} ({artefact.stat().st_size / 1024:.1f} KB)")
    print("  This is the file the AI engineer uploads to deploy a new model.")

    print()
    print("Demo complete.")


if __name__ == "__main__":
    sys.exit(main())
