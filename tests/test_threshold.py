"""Guards on the two decisions a metrics table cannot check.

A cutoff that flags the wrong share of the book, or a feature collapse that
quietly drops a level, produces output that looks entirely plausible and is
wrong for every customer. Neither shows up in ROC-AUC.
"""
import sys; sys.path.insert(0, '.')

import numpy as np
import pytest

from src.features import PAY_LEVELS, collapse_codes, collapse_pay_code
from src.threshold import capacity_threshold, threshold_sweep

rng = np.random.default_rng(0)
P = rng.beta(2, 6, size=5000)                       # scores skewed toward safe
Y = rng.binomial(1, np.clip(P * 1.6, 0, 1))         # defaults correlated with them


# ----------------------------------------------------------------- capacity

@pytest.mark.parametrize("capacity", [0.05, 0.10, 0.20, 0.50])
def test_capacity_threshold_flags_that_share(capacity):
    """The whole point of the policy: the queue is the size the team can work."""
    t = capacity_threshold(P, capacity)
    assert (P >= t).mean() == pytest.approx(capacity, abs=0.01)


def test_a_smaller_queue_needs_a_higher_bar():
    assert capacity_threshold(P, 0.05) > capacity_threshold(P, 0.20)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_impossible_capacities_are_rejected(bad):
    with pytest.raises(ValueError):
        capacity_threshold(P, bad)


# ---------------------------------------------------------------- the sweep

def test_precision_and_recall_move_in_opposite_directions():
    """The trade-off the cutoff is chosen against.

    If both moved the same way there would be a free lunch, and the threshold
    would not be a business decision at all.
    """
    s = threshold_sweep(Y, P).dropna(subset=["precision"])
    assert s.recall.is_monotonic_decreasing
    assert s.precision.iloc[-1] > s.precision.iloc[0]
    assert s.flagged_rate.is_monotonic_decreasing


def test_lift_is_above_one_where_the_model_has_signal():
    s = threshold_sweep(Y, P).dropna(subset=["lift"])
    assert (s.lift > 1).all()


# ------------------------------------------------------------- the collapse

@pytest.mark.parametrize("code", [3, 4, 5, 6, 7, 8, 9])
def test_high_repayment_codes_are_pooled(code):
    """Codes >= 3 hold too few accounts to support their own cell."""
    assert collapse_pay_code(code) == "3+"


@pytest.mark.parametrize("code", [-2, -1, 0, 1, 2])
def test_documented_codes_keep_their_own_level(code):
    assert collapse_pay_code(code) == str(code)


def test_collapse_produces_only_known_levels():
    import pandas as pd

    frame = pd.DataFrame({
        "PAY_0": [-2, -1, 0, 1, 2, 3, 7], "PAY_2": [0] * 7, "PAY_3": [0] * 7,
        "PAY_4": [0] * 7, "PAY_5": [0] * 7, "PAY_6": [0] * 7,
        "EDUCATION": [1, 2, 3, 4, 5, 6, 0], "MARRIAGE": [1, 2, 3, 0, 1, 2, 3],
        "SEX": [1, 2, 1, 2, 1, 2, 1],
    })
    out = collapse_codes(frame)
    assert set(out.PAY_0) <= set(PAY_LEVELS)
    assert set(out.EDUCATION) == {"graduate", "university", "high_school", "other"}
    assert set(out.MARRIAGE) == {"married", "single", "other"}
    assert out.isna().sum().sum() == 0


def test_collapse_does_not_mutate_its_input():
    """It runs inside a Pipeline on every CV fold; mutating would corrupt later folds."""
    import pandas as pd

    frame = pd.DataFrame({
        "PAY_0": [5], "PAY_2": [0], "PAY_3": [0], "PAY_4": [0], "PAY_5": [0],
        "PAY_6": [0], "EDUCATION": [1], "MARRIAGE": [1], "SEX": [1],
    })
    collapse_codes(frame)
    assert frame.PAY_0.iloc[0] == 5
