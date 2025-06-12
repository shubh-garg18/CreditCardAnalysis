"""Choosing where to cut the probability ranking.

The model outputs a probability. A bank needs a decision. Turning one into the
other is a separate choice, and it is a business choice, not a modelling one:

    P(default) = 0.31   is an *estimate*
    "send to review"    is a *decision*

0.5 is the default cutoff in every library, and it is the right cutoff only when
a false positive and a false negative cost the same. In collections they do not:
missing a defaulter costs a share of the balance, while a false alarm costs a
phone call. But the opposite extreme -- "a miss is 10x worse, so flag anyone
above 0.09" -- flags most of the book, and no review team can work that queue.

So the cutoff here is set by capacity: review the riskiest 20% of accounts. The
20% is a stated assumption, and `threshold_sweep` shows what the alternatives
would have cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import threshold_metrics


def threshold_sweep(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Every threshold metric across a grid of cutoffs.

    This is the table behind the precision/recall trade-off: as the cutoff
    rises, precision rises and recall falls. Reading it is how the operating
    point gets argued rather than assumed.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return pd.DataFrame([threshold_metrics(y_true, y_proba, t) for t in thresholds])


def capacity_threshold(y_proba, capacity: float) -> float:
    """The cutoff that flags exactly `capacity` share of the scored population.

    `capacity=0.20` returns the 80th percentile of the predicted probabilities,
    so the 20% riskiest accounts fall above it. Expressing the policy as "the
    worst 20%" rather than "p >= 0.34" means it survives a model refit: the
    queue stays the size the team can actually work.
    """
    if not 0 < capacity < 1:
        raise ValueError(f"capacity must be in (0, 1), got {capacity}")
    return float(np.quantile(np.asarray(y_proba), 1.0 - capacity))
