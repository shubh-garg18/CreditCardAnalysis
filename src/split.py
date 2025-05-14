"""Split design.

Three-way stratified split: train (60%) / validation (20%) / test (20%).

* **train** — fit models.
* **validation** — model selection, hyperparameters, threshold.
* **test** — touched once, at the end, to report final numbers.

**No temporal split.** See `why_not_temporal()`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import TARGET, ROOT, load_uci

PROCESSED = ROOT / "data" / "processed"
MANIFEST = PROCESSED / "split_manifest.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # of the full dataset, taken from the non-test remainder


def why_not_temporal() -> str:
    """Why this dataset cannot support a temporal split, stated explicitly.

    A temporal split needs rows that can be ordered in time. Here every customer
    is observed over the *same* calendar window -- April to September 2005 -- with
    one row each and a single shared outcome month, October 2005. There is no
    column that orders customers in time, no cohort structure, and no second
    snapshot of anyone.

    For a real behaviour scorecard a temporal (out-of-time) split is the correct
    design: you train on older cohorts and validate on newer ones, because it is
    the only way to measure whether the model survives population drift. That is
    what a bank would do. We cannot, and inventing a proxy -- ordering by ID, say
    -- would fabricate a validation property the data does not have.

    Consequence to state in the write-up: these estimates measure generalisation
    to *unseen customers*, not to a *future time period*.
    """
    return why_not_temporal.__doc__


def _frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def make_splits(force: bool = False) -> dict[str, pd.DataFrame]:
    """Deduplicate, then split stratified on the target, then persist.

    Deduplication happens **before** the split, not after. 56 rows in UCI are
    exact copies of another row; if a row and its twin land on opposite sides,
    the model is scored on data it memorised. Order matters: deduplicating after splitting
    would leave the cross-split pairs intact.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and not force:
        return {n: load_split(n) for n in ("train", "val", "test")}

    uci = load_uci()
    features = [c for c in uci.columns if c != TARGET]

    n_before = len(uci)
    deduped = uci[~uci.duplicated(subset=features, keep="first")]
    n_dropped = n_before - len(deduped)

    y = deduped[TARGET]
    rest, test = train_test_split(
        deduped, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # VAL_SIZE is a share of the full set, so rescale against what remains.
    rest, val = train_test_split(
        rest,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=rest[TARGET],
        random_state=RANDOM_STATE,
    )
    splits = {"train": rest, "val": val, "test": test}

    for name, df in splits.items():
        df.to_parquet(PROCESSED / f"{name}.parquet")

    MANIFEST.write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "source": "UCI default of credit card clients",
        "rows_before_dedup": n_before,
        "duplicate_rows_dropped": n_dropped,
        "random_state": RANDOM_STATE,
        "splits": {
            name: {
                "n": len(df),
                "share": round(len(df) / len(deduped), 4),
                "default_rate": round(float(df[TARGET].mean()), 6),
                "sha256_16": _frame_hash(df),
            }
            for name, df in splits.items()
        },
    }, indent=2))
    return splits


def load_split(name: str) -> pd.DataFrame:
    """Load one persisted split."""
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {name!r}")
    path = PROCESSED / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run make_splits() first")
    return pd.read_parquet(path)


def split_summary() -> pd.DataFrame:
    """The table that goes in the README."""
    manifest = json.loads(MANIFEST.read_text())
    return pd.DataFrame(manifest["splits"]).T
