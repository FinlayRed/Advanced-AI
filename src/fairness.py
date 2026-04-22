"""Producer-fairness re-ranking for the Bristol Food Network recommender.

The case study explicitly calls out the risk of "favouring certain producers
in recommendations". When the same fruit or vegetable is offered by several
producers a naive recommender will tend to lock onto whichever producer has
the most data, freezing newer or smaller producers out of the marketplace.

This module post-processes a ranked list with a simple round-robin penalty:
each time a producer appears in the list, subsequent items from the same
producer are demoted by a configurable factor. The catalogue size in this
project is small enough that more sophisticated fairness algorithms (e.g.
FA*IR, equity of attention) would be overkill - this one is easy to explain
in the demo and easy to ablate in the report.
"""

from __future__ import annotations

import pandas as pd


def rerank_for_producer_diversity(
    ranked_product_ids: list[str],
    products: pd.DataFrame,
    penalty: float = 0.5,
    k: int | None = None,
) -> list[str]:
    """Re-rank a candidate list to spread recommendations across producers.

    Parameters
    ----------
    ranked_product_ids:
        The original ranking from the underlying recommender, best first.
        Should contain more candidates than ``k`` so re-ranking has room to
        work - typically pass top 3K candidates and ask for top K.
    products:
        Product catalogue, must contain ``product_id`` and ``producer_id``.
    penalty:
        Multiplier applied to an item's effective score for each previous
        appearance of its producer in the output list. ``penalty=1.0``
        disables fairness; ``penalty=0.0`` enforces strict round-robin.
    k:
        Optional cap on the output length. Defaults to the input length.
    """
    product_to_producer = dict(zip(products["product_id"], products["producer_id"]))
    n = len(ranked_product_ids)
    if k is None:
        k = n

    # Original implicit scores - we only need the relative ordering, so map
    # rank position to a decreasing score in [0, 1].
    base_scores = {
        pid: 1.0 - (i / max(n, 1))
        for i, pid in enumerate(ranked_product_ids)
    }

    selected: list[str] = []
    producer_uses: dict[str, int] = {}
    remaining = set(ranked_product_ids)

    while len(selected) < k and remaining:
        best_pid = None
        best_score = -1.0
        for pid in remaining:
            producer = product_to_producer.get(pid, "_unknown")
            adjusted = base_scores[pid] * (penalty ** producer_uses.get(producer, 0))
            if adjusted > best_score:
                best_score = adjusted
                best_pid = pid
        assert best_pid is not None
        selected.append(best_pid)
        remaining.discard(best_pid)
        producer = product_to_producer.get(best_pid, "_unknown")
        producer_uses[producer] = producer_uses.get(producer, 0) + 1

    return selected
