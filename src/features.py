"""Feature preparation.

Everything here is **stateless**: fixed maps chosen from the EDA in notebooks 01
and 02, with no quantity estimated from data. That matters for leakage --
a stateless transform cannot carry information across a split. Anything that does
learn from data (scaling, imputation) lives in the sklearn pipeline instead, so it
is fit on train and merely applied elsewhere.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BILL_COLS, PAY_AMT_COLS, PAY_COLS

# Repayment codes. Notebook 01 established that these are nominal, not ordinal:
# default rate runs 0 < -2 < -1 << 1 << 2, so the integers do not order risk.
# Codes >=3 are pooled -- individually they hold between 1 and 210 training rows
# per column, with non-monotonic rates, which is sampling noise rather than a
# risk gradient.
PAY_LEVELS = ["-2", "-1", "0", "1", "2", "3+"]

# EDUCATION: 1/2/3 are documented. 0, 5, 6 are undefined in the codebook and 4 is
# "others"; pooled they are 275 training rows at 5.8%, well below every
# documented level, so they carry signal but cannot support four separate cells.
EDUCATION_MAP = {1: "graduate", 2: "university", 3: "high_school",
                 0: "other", 4: "other", 5: "other", 6: "other"}

# MARRIAGE: 0 is undefined, 3 is "others". Pooled they sit at 22.4%, which is
# indistinguishable from "married" (22.9%) -- expect this feature to be weak.
MARRIAGE_MAP = {1: "married", 2: "single", 0: "other", 3: "other"}

SEX_MAP = {1: "male", 2: "female"}

CATEGORICAL = PAY_COLS + ["EDUCATION", "MARRIAGE", "SEX"]
NUMERIC = ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_AMT_COLS


def collapse_pay_code(value: int) -> str:
    """Map a repayment status to its collapsed level."""
    if value >= 3:
        return "3+"
    return str(int(value))


def collapse_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every categorical collapse. Stateless, so leakage-safe anywhere."""
    out = df.copy()
    for col in PAY_COLS:
        out[col] = out[col].map(collapse_pay_code).astype("object")
    out["EDUCATION"] = out["EDUCATION"].map(EDUCATION_MAP).fillna("other")
    out["MARRIAGE"] = out["MARRIAGE"].map(MARRIAGE_MAP).fillna("other")
    out["SEX"] = out["SEX"].map(SEX_MAP)
    return out


def build_preprocessor(scale: bool = True, drop_sex: bool = False) -> ColumnTransformer:
    """Column-wise preprocessing.

    `scale` is True for models with a linear decision surface and False for trees,
    which only read rank order. Making it a parameter rather than a global
    keeps the choice explicit per model instead of applied blindly to all of them.

    `drop_sex` excludes the protected attribute from the design matrix. The
    final model sets it, because sex cannot lawfully enter a credit decision in
    most jurisdictions.

    `handle_unknown='infrequent_if_exist'` matters when scoring new data: a
    category absent from training must not raise. Encoding each split separately
    and patching the columns afterwards fails silently when a category is
    missing from one of them.
    """
    categorical = [c for c in CATEGORICAL if not (drop_sex and c == "SEX")]
    numeric_steps = [("scale", StandardScaler())] if scale else [("passthrough", "passthrough")]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=20, sparse_output=False),
             categorical),
            ("num", Pipeline(numeric_steps) if scale else "passthrough", NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(fitted: ColumnTransformer) -> list[str]:
    return list(fitted.get_feature_names_out())
