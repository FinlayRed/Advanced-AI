"""Produce the charts used in the demo slides and the technical report.

Run from the project root:

    python -m src.charts

Writes PNG figures to ``reports/figures/``. Three charts are generated:

1. ``metric_comparison.png`` - bar chart of Precision/Recall/Hit/NDCG @ 5
   and @ 10 for the baseline vs the NMF model. The headline result.
2. ``producer_concentration.png`` - share of recommendation slots per
   producer, with and without the fairness re-ranker. Evidence of the
   fairness mitigation working.
3. ``seasonality.png`` - monthly demand for two contrasting items
   (Strawberry vs Apple) showing the seasonal patterns the recommender
   has to learn from. Supports the problem-characterisation section.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baseline import FrequencyRecencyRecommender
from src.data_generator import generate
from src.evaluation import (
    chronological_split,
    evaluate,
    producer_concentration,
)
from src.fairness import rerank_for_producer_diversity
from src.mf_model import NMFRecommender


FIGURES_DIR = Path("reports/figures")


def _ensure_data() -> dict:
    """Generate the dataset if it isn't already on disk."""
    data_dir = Path("data")
    if (data_dir / "orders.csv").exists():
        return {
            "customers": pd.read_csv(data_dir / "customers.csv"),
            "products": pd.read_csv(data_dir / "products.csv"),
            "orders": pd.read_csv(data_dir / "orders.csv"),
        }
    return generate(data_dir)


def chart_metric_comparison(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Bar chart of every metric at K=5 and K=10 for both models."""
    baseline = FrequencyRecencyRecommender().fit(train)
    nmf = NMFRecommender(n_components=12).fit(train)
    b = evaluate(baseline, train, test, k_values=(5, 10))
    n = evaluate(nmf, train, test, k_values=(5, 10))
    merged = b.merge(n, on=["k", "metric"], suffixes=("_baseline", "_nmf"))
    merged["label"] = merged["metric"].str.title() + "@" + merged["k"].astype(str)
    merged = merged.sort_values(["k", "metric"]).reset_index(drop=True)

    x = np.arange(len(merged))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, merged["score_baseline"], width,
           label="Baseline (Freq×Recency)", color="#7a8aa1")
    ax.bar(x + width / 2, merged["score_nmf"], width,
           label="NMF (k=12 latent factors)", color="#2a6f97")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["label"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recommendation quality: Baseline vs NMF (chronological 80/20 split)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i, (b_val, n_val) in enumerate(zip(merged["score_baseline"], merged["score_nmf"])):
        ax.text(i - width / 2, b_val + 0.015, f"{b_val:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, n_val + 0.015, f"{n_val:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metric_comparison.png", dpi=160)
    plt.close(fig)


def chart_producer_concentration(train: pd.DataFrame, products: pd.DataFrame) -> None:
    """Share of recommendation slots per producer, with and without fairness."""

    class _RerankWrapper:
        """Tiny shim that applies fairness re-ranking on top of NMF outputs."""

        def __init__(self, base: NMFRecommender, products: pd.DataFrame, penalty: float):
            self._base = base
            self._products = products
            self._penalty = penalty

        def fit(self, orders):
            return self

        def recommend_next_order(self, customer_id: str, k: int) -> list[str]:
            cands = self._base.recommend_next_order(
                customer_id, k=min(k * 3, len(self._products))
            )
            return rerank_for_producer_diversity(
                cands, self._products, penalty=self._penalty, k=k
            )

    nmf = NMFRecommender(n_components=12).fit(train)
    fair = _RerankWrapper(nmf, products, penalty=0.6)
    cohort = train["customer_id"].value_counts().head(200).index.tolist()
    raw = producer_concentration(nmf, cohort, products, k=10)
    fair_share = producer_concentration(fair, cohort, products, k=10)

    merged = raw.merge(fair_share, on="producer_id",
                       suffixes=("_raw", "_fair"), how="outer").fillna(0.0)
    merged = merged.sort_values("share_raw", ascending=False).reset_index(drop=True)

    x = np.arange(len(merged))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, merged["share_raw"] * 100, width,
           label="Without fairness re-ranking", color="#c0504d")
    ax.bar(x + width / 2, merged["share_fair"] * 100, width,
           label="With fairness re-ranking (penalty=0.6)", color="#5b9b56")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["producer_id"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share of recommendation slots (%)")
    ax.set_title("Producer concentration in top-10 recommendations (200-customer cohort)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "producer_concentration.png", dpi=160)
    plt.close(fig)


def chart_seasonality(orders: pd.DataFrame) -> None:
    """Monthly demand for two contrasting categories - shows the patterns
    the recommender has to exploit."""
    o = orders.copy()
    o["order_timestamp"] = pd.to_datetime(o["order_timestamp"])
    o["month"] = o["order_timestamp"].dt.month
    by_month = (
        o[o["category"].isin(["Strawberry", "Apple"])]
        .groupby(["month", "category"])["quantity_kg"].sum().unstack()
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    by_month.plot(ax=ax, marker="o", linewidth=2,
                  color={"Strawberry": "#c0392b", "Apple": "#27ae60"})
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Total kg sold (across both years)")
    ax.set_xlabel("")
    ax.set_title("Seasonal demand patterns the recommender must exploit")
    ax.grid(alpha=0.3)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "seasonality.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = _ensure_data()
    train, test = chronological_split(data["orders"], test_fraction=0.2)
    chart_metric_comparison(train, test)
    chart_producer_concentration(train, data["products"])
    chart_seasonality(data["orders"])
    for f in sorted(FIGURES_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  Wrote {f}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
