"""Evaluation.

One module computes every number this project reports, so there is exactly one
place where a metric could be wrong -- and `tests/test_data.py` pins the fixed
points (a perfect ranker must score exactly 1.0).

Nothing here rounds, fills in, or adjusts. If a quantity is undefined (precision
with no positive predictions) it comes back as NaN rather than 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def threshold_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Metrics that depend on where the cutoff is placed."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    flagged = tp + fp
    base = float(np.mean(y_true))
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision_score(y_true, y_pred, zero_division=np.nan),
        "recall": recall_score(y_true, y_pred, zero_division=np.nan),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=np.nan),
        "flagged_rate": flagged / len(y_true),
        # How much better than flagging the same number of people at random.
        "lift": (tp / flagged) / base if flagged else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def ranking_metrics(y_true, y_proba) -> dict:
    """Metrics over the whole probability ranking, independent of any threshold.

    PR-AUC is reported alongside its own floor: unlike ROC-AUC, whose
    no-skill value is always 0.5, a no-skill PR-AUC equals the base rate. Quoting
    PR-AUC without that floor is meaningless.
    """
    base = float(np.mean(y_true))
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "pr_auc_no_skill": base,
        "brier": brier_score_loss(y_true, y_proba),
    }


def evaluate(y_true, y_proba, threshold: float = 0.5, label: str = "") -> dict:
    out = {"model": label} if label else {}
    out.update(ranking_metrics(y_true, y_proba))
    out.update(threshold_metrics(y_true, y_proba, threshold))
    return out


def majority_baseline(y_true) -> dict:
    """The model that always predicts the majority class.

    Any candidate that fails to beat this on accuracy has learned nothing, and
    quoting accuracy without this number alongside is how 78% gets called good.
    """
    y_true = np.asarray(y_true)
    base = y_true.mean()
    return {
        "model": "majority class (predict 'no default')",
        "accuracy": max(base, 1 - base),
        "precision": np.nan, "recall": 0.0, "f1": 0.0,
        "roc_auc": 0.5, "pr_auc": base, "pr_auc_no_skill": base,
    }


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    cols = ["model", "roc_auc", "pr_auc", "pr_auc_no_skill", "brier",
            "accuracy", "precision", "recall", "specificity", "f1",
            "flagged_rate", "lift", "threshold"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]
