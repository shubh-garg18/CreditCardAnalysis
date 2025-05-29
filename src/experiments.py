"""Experiment log.

Every model that gets fit appends a row here with its configuration and its
scores. The log is written by code, never edited by hand, so the comparison
table in the README is a query rather than a retyping.

That way every number in the write-up can be traced back to the run that
produced it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import ROOT

LOG = ROOT / "experiments" / "log.jsonl"


def record(experiment: str, model: str, params: dict, metrics: dict,
           split: str = "val", notes: str = "") -> dict:
    """Append one run. Returns the row written."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
        "model": model,
        "split": split,
        "params": json.dumps(params, sort_keys=True, default=str),
        "notes": notes,
        **{k: v for k, v in metrics.items() if k != "model"},
    }
    with LOG.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def load() -> pd.DataFrame:
    if not LOG.exists():
        return pd.DataFrame()
    return pd.read_json(LOG, lines=True)


def leaderboard(metric: str = "pr_auc", split: str = "val", top: int | None = None) -> pd.DataFrame:
    """Runs ranked by one metric.

    Deliberately shows several metrics at once: picking a final model on a single
    number is how you end up with the best ranker rather than the most useful
    model.
    """
    df = load()
    if df.empty:
        return df
    df = df[df["split"] == split].sort_values(metric, ascending=False)
    cols = [c for c in ["experiment", "model", "roc_auc", "pr_auc", "brier",
                        "precision", "recall", "f1", "threshold", "notes"]
            if c in df.columns]
    return (df[cols].head(top) if top else df[cols]).reset_index(drop=True)


def clear() -> None:
    """Reset the log. Used when a notebook is re-run end to end so repeated
    executions do not stack duplicate rows."""
    if LOG.exists():
        LOG.unlink()
