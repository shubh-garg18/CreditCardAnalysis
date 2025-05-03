"""Data loading and integrity checks.

Modelling runs on **UCI** `default of credit card clients` (Yeh & Lien, 2009):
30,000 Taiwanese credit-card accounts observed April-September 2005, with the
October 2005 repayment outcome as the target.

The Finance Club distributed its own copy of the same data with the validation
labels withheld; `load_finclub_validation` reads that file so the project can
produce the required prediction CSV.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

UCI_XLS = RAW / "uci_default_of_credit_card_clients.xls"
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)

TARGET = "default_next_month"

# PAY_1 does not exist in the source; the September column is named PAY_0.
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHIC_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]

# Column index 1 is the most recent month. PAY_0/BILL_AMT1/PAY_AMT1 are
# September 2005; the target is the October 2005 outcome.
MONTH_OF_INDEX = {
    1: "2005-09", 2: "2005-08", 3: "2005-07",
    4: "2005-06", 5: "2005-05", 6: "2005-04",
}


def download_uci(dest: Path = UCI_XLS) -> Path:
    """Fetch the UCI archive if it is not already on disk.

    Keeps the repository reproducible from a clean clone without committing
    the data itself.
    """
    if dest.exists():
        return dest
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xls"))
        dest.write_bytes(zf.read(name))
    return dest


def load_uci(path: Path = UCI_XLS) -> pd.DataFrame:
    """Load UCI with the header row fixed and the target renamed.

    The .xls carries a two-level header: row 0 holds `X1..X23`, row 1 holds the
    real names. `header=1` selects the latter.
    """
    download_uci(path)
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"default payment next month": TARGET})
    df = df.set_index("ID")
    return df


# Finclub renamed and lower-cased some columns, and recoded sex as 0/1.
FINCLUB_RENAME = {
    "age": "AGE", "sex": "SEX", "education": "EDUCATION", "marriage": "MARRIAGE",
    **{f"pay_{i}": f"PAY_{i}" for i in (0, 2, 3, 4, 5, 6)},
    **{f"Bill_amt{i}": f"BILL_AMT{i}" for i in range(1, 7)},
    **{f"pay_amt{i}": f"PAY_AMT{i}" for i in range(1, 7)},
}
# Finclub 0 = male, 1 = female; UCI uses 1 = male, 2 = female.
FINCLUB_SEX_MAP = {0: 1, 1: 2}
# Distributed by the Finance Club, not public, so it is not in the repository.
FINCLUB_VALIDATION = RAW / "validate_dataset_final.csv"


def load_finclub_validation() -> pd.DataFrame:
    """Load the unlabelled validation set the Finance Club asks predictions for.

    Returns the 5,016 customers with UCI column names, so the same fitted
    pipeline scores them without a second copy of the preprocessing.

    `AVG_Bill_amt` and `PAY_TO_BILL_ratio` are dropped: they are deterministic
    functions of columns the model already has, and the model was never trained
    on them.
    """
    df = pd.read_csv(FINCLUB_VALIDATION)
    df = df.rename(columns=FINCLUB_RENAME)
    df["SEX"] = df["SEX"].map(FINCLUB_SEX_MAP)
    if df["SEX"].isna().any():
        raise ValueError("unexpected sex code in the Finclub validation file")
    keep = ["Customer_ID"] + DEMOGRAPHIC_COLS + PAY_COLS + BILL_COLS + PAY_AMT_COLS
    return df[keep]


# --------------------------------------------------------------------------
# Inspection helpers used by the EDA notebook.
# --------------------------------------------------------------------------


def rate_by_code(df: pd.DataFrame, col: str, target: str = TARGET) -> pd.DataFrame:
    """Default rate per distinct value, with a Wilson 95% interval.

    The interval matters here: several repayment codes have single-digit
    counts, and a point estimate alone invites reading a trend into noise.
    """
    g = df.groupby(col)[target].agg(n="size", defaults="sum", rate="mean")
    lo, hi = wilson_interval(g["defaults"], g["n"])
    g["ci_lo"], g["ci_hi"] = lo, hi
    g["share"] = g["n"] / len(df)
    return g


def wilson_interval(successes, n, z: float = 1.96):
    """Wilson score interval -- behaves sensibly when n is small or rate is 0/1,
    unlike the normal approximation."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - margin, centre + margin


def duplicate_report(df: pd.DataFrame, subset: list[str] | None = None) -> dict:
    """Count rows that are not unique on `subset` (default: all columns)."""
    cols = subset if subset is not None else list(df.columns)
    dup_mask = df.duplicated(subset=cols, keep=False)
    return {
        "n_rows": len(df),
        "n_rows_in_dup_groups": int(dup_mask.sum()),
        "n_redundant_rows": int(df.duplicated(subset=cols, keep="first").sum()),
        "n_groups": int(df[dup_mask].groupby(cols, observed=True).ngroups) if dup_mask.any() else 0,
    }
