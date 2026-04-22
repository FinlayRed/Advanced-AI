"""Synthetic purchase-history generator for the Bristol Regional Food Network.

Produces customer, product, producer, and order tables that mirror the kind of
data the live marketplace would accumulate. Categories deliberately match the
fruit/vegetable classes used by the CV quality model (Task 2) so the two
subsystems demo as one coherent platform.

The generator builds in three structural patterns the recommender needs to
exploit and be evaluated against:

1. Recurring household staples - most customers re-order a small basket of
   "favourite" items repeatedly (the long tail of one-off buys is also present).
2. UK seasonality - strawberries peak in summer, apples in autumn, etc.
   Demand multipliers per month are documented in ``SEASONALITY``.
3. Producer competition - every SKU is offered by 2-4 producers, so the
   recommender's producer-fairness behaviour can be measured.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Categories taken directly from the Kaggle fresh/rotten dataset Yas is using
# for the Task 2 CV model. Keeping these aligned means the two subsystems can
# share a single product catalogue at integration time.
PRODUCE_CATEGORIES: list[str] = [
    "Apple", "Banana", "Bellpepper", "Carrot", "Cucumber",
    "Grape", "Guava", "Jujube", "Mango", "Orange",
    "Pomegranate", "Potato", "Strawberry", "Tomato",
]

# Monthly demand multipliers reflecting roughly typical UK seasonal patterns.
# Values are multiplicative weights applied to baseline demand. They do not
# need to be perfectly accurate - the goal is for the recommender's seasonal
# behaviour to be observable and testable.
SEASONALITY: dict[str, list[float]] = {
    # Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
    "Apple":       [1.1, 1.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.1, 1.4, 1.6, 1.5, 1.3],
    "Strawberry":  [0.4, 0.4, 0.5, 0.7, 1.4, 1.8, 1.9, 1.5, 0.9, 0.6, 0.4, 0.4],
    "Tomato":      [0.7, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 1.5, 1.2, 1.0, 0.8, 0.7],
    "Pomegranate": [1.3, 1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.8, 1.0, 1.4, 1.5, 1.4],
    "Mango":       [0.7, 0.8, 0.9, 1.1, 1.3, 1.4, 1.4, 1.3, 1.1, 0.9, 0.8, 0.7],
    "Grape":       [0.8, 0.8, 0.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.2, 1.0, 0.9],
    "Orange":      [1.4, 1.3, 1.1, 1.0, 0.9, 0.8, 0.8, 0.8, 0.9, 1.0, 1.2, 1.4],
    "Bellpepper":  [0.9, 0.9, 1.0, 1.0, 1.1, 1.2, 1.2, 1.2, 1.1, 1.0, 0.9, 0.9],
    "Cucumber":    [0.8, 0.8, 0.9, 1.0, 1.2, 1.3, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8],
    "Carrot":      [1.1, 1.1, 1.0, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2, 1.2],
    "Potato":      [1.2, 1.1, 1.0, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2, 1.2],
    "Banana":      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Guava":       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0],
    "Jujube":      [0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.2, 1.1, 1.0],
}

# Baseline popularity weights - some items are intrinsically more popular than
# others regardless of season (apples and bananas are everyday staples, jujube
# is niche). Used when sampling which items go into a customer's "favourites".
BASE_POPULARITY: dict[str, float] = {
    "Apple": 1.6, "Banana": 1.6, "Tomato": 1.4, "Potato": 1.4,
    "Carrot": 1.3, "Cucumber": 1.2, "Orange": 1.2, "Bellpepper": 1.1,
    "Grape": 1.0, "Strawberry": 1.0, "Mango": 0.8, "Pomegranate": 0.6,
    "Guava": 0.5, "Jujube": 0.3,
}


@dataclass(frozen=True)
class GeneratorConfig:
    """Tunable parameters for the synthetic data generator."""

    n_customers: int = 800
    n_producers: int = 12
    start_date: datetime = datetime(2024, 1, 1)
    end_date: datetime = datetime(2025, 12, 31)
    avg_orders_per_customer: float = 25.0
    seed: int = 42


def _build_product_catalogue(
    config: GeneratorConfig,
    rng: random.Random,
) -> pd.DataFrame:
    """Create the SKU catalogue. Each produce type is sold by 2-4 producers."""
    rows: list[dict] = []
    product_id = 0
    for category in PRODUCE_CATEGORIES:
        n_offering = rng.randint(2, 4)
        offering_producers = rng.sample(range(config.n_producers), n_offering)
        for producer_id in offering_producers:
            # Per-kg price varies by item and producer to make ranking
            # decisions non-trivial when fairness re-ranking is applied.
            base_price = rng.uniform(1.2, 4.5)
            rows.append({
                "product_id": f"P{product_id:04d}",
                "category": category,
                "producer_id": f"PR{producer_id:03d}",
                "price_per_kg": round(base_price, 2),
            })
            product_id += 1
    return pd.DataFrame(rows)


def _build_customers(
    config: GeneratorConfig,
    rng: random.Random,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each customer 3-7 favourite categories and an order frequency."""
    rows: list[dict] = []
    for i in range(config.n_customers):
        n_favourites = rng.randint(3, 7)
        # Weighted sample of favourite categories using base popularity, so
        # popular items appear in more customers' favourite sets.
        favourites = rng.choices(
            PRODUCE_CATEGORIES,
            weights=[BASE_POPULARITY[c] for c in PRODUCE_CATEGORIES],
            k=n_favourites * 3,  # over-sample then de-duplicate
        )
        favourites = list(dict.fromkeys(favourites))[:n_favourites]
        # Per-customer order frequency varies by a factor of ~5x.
        rate_factor = np.clip(rng.gauss(1.0, 0.4), 0.3, 2.5)
        rows.append({
            "customer_id": f"C{i:04d}",
            "favourite_categories": ",".join(favourites),
            "order_rate_factor": round(rate_factor, 3),
            "signup_date": (
                config.start_date
                + timedelta(days=rng.randint(0, 200))
            ).date().isoformat(),
        })
    return pd.DataFrame(rows)


def _generate_orders(
    config: GeneratorConfig,
    rng: random.Random,
    np_rng: np.random.Generator,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the order log.

    Each customer's purchase stream is sampled per-day with a probability that
    depends on (a) their personal order rate and (b) the seasonal demand
    multiplier averaged across their favourite categories. When a customer
    places an order they buy 2-6 items, drawn ~70% from favourites (with
    season-adjusted weights) and ~30% from the wider catalogue (exploration).
    """
    days = (config.end_date - config.start_date).days
    # Daily probability scaled so that an average customer hits roughly the
    # configured avg_orders_per_customer over the time window.
    base_daily_p = config.avg_orders_per_customer / days

    cat_to_products: dict[str, list[str]] = {
        c: products.loc[products["category"] == c, "product_id"].tolist()
        for c in PRODUCE_CATEGORIES
    }
    all_product_ids = products["product_id"].tolist()
    product_to_category = dict(zip(products["product_id"], products["category"]))

    orders: list[dict] = []
    order_counter = 0

    for _, cust in customers.iterrows():
        favourites = cust["favourite_categories"].split(",")
        signup = datetime.fromisoformat(cust["signup_date"])
        active_days = (config.end_date - signup).days
        daily_p = base_daily_p * cust["order_rate_factor"]

        for d in range(active_days):
            current_date = signup + timedelta(days=d)
            month_idx = current_date.month - 1
            seasonal_multiplier = float(np.mean([
                SEASONALITY[c][month_idx] for c in favourites
            ]))
            if np_rng.random() > daily_p * seasonal_multiplier:
                continue

            n_items = rng.randint(2, 6)
            order_id = f"O{order_counter:06d}"
            order_counter += 1

            for _ in range(n_items):
                if rng.random() < 0.7:
                    # Buy from favourites, weighted by current seasonality.
                    weights = [SEASONALITY[c][month_idx] for c in favourites]
                    chosen_cat = rng.choices(favourites, weights=weights, k=1)[0]
                else:
                    # Exploration buy from anywhere in the catalogue.
                    chosen_cat = rng.choices(
                        PRODUCE_CATEGORIES,
                        weights=[BASE_POPULARITY[c] for c in PRODUCE_CATEGORIES],
                        k=1,
                    )[0]
                product_id = rng.choice(cat_to_products[chosen_cat])
                quantity_kg = round(rng.uniform(0.25, 3.0), 2)
                orders.append({
                    "order_id": order_id,
                    "customer_id": cust["customer_id"],
                    "product_id": product_id,
                    "category": product_to_category[product_id],
                    "quantity_kg": quantity_kg,
                    "order_timestamp": current_date.isoformat(),
                })

    return pd.DataFrame(orders)


def generate(
    output_dir: Path | str = "data",
    config: GeneratorConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset and write CSVs to ``output_dir``.

    Returns the three dataframes keyed by ``customers``, ``products``,
    ``orders`` so callers can use them in-process without re-reading from disk.
    """
    if config is None:
        config = GeneratorConfig()

    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    products = _build_product_catalogue(config, rng)
    customers = _build_customers(config, rng, products)
    orders = _generate_orders(config, rng, np_rng, customers, products)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    products.to_csv(output_dir / "products.csv", index=False)
    customers.to_csv(output_dir / "customers.csv", index=False)
    orders.to_csv(output_dir / "orders.csv", index=False)

    return {"customers": customers, "products": products, "orders": orders}


if __name__ == "__main__":
    data = generate()
    print(f"Customers: {len(data['customers']):,}")
    print(f"Products:  {len(data['products']):,}")
    print(f"Orders:    {len(data['orders']):,}")
    print(f"Unique categories: {data['products']['category'].nunique()}")
    print(f"Unique producers:  {data['products']['producer_id'].nunique()}")
