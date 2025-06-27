"""Guards on the data layer and on the fixed points of the metrics."""
import sys; sys.path.insert(0, '.')
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.data import load_uci, rate_by_code, wilson_interval, TARGET, PAY_COLS


@pytest.fixture(scope="module")
def uci():
    return load_uci()


def test_shape_and_target(uci):
    assert uci.shape == (30000, 24)
    assert set(uci[TARGET].unique()) == {0, 1}
    assert uci.isna().sum().sum() == 0


def test_base_rate_is_what_we_report(uci):
    # Every "22%" claim in the project traces here.
    assert uci[TARGET].mean() == pytest.approx(0.2212, abs=5e-5)


def test_pay_1_does_not_exist(uci):
    # The September column is PAY_0. Assuming PAY_1 exists silently drops a month.
    assert "PAY_1" not in uci.columns
    assert all(c in uci.columns for c in PAY_COLS)


def test_pay_0_is_not_ordinal(uci):
    """The finding that drives the encoding decision.

    If PAY_0 were ordinal, default rate would rise monotonically with the code.
    Code 0 is safer than -1, so it does not.
    """
    r = uci.groupby("PAY_0")[TARGET].mean()
    assert r.loc[0] < r.loc[-1], "code 0 should be safer than -1"
    assert not r.loc[[-2, -1, 0]].is_monotonic_increasing


def test_negative_bills_survive_loading(uci):
    # The Finclub copy clipped these to zero, erasing a real low-risk signal.
    assert (uci[[f"BILL_AMT{i}" for i in range(1, 7)]] < 0).values.sum() > 0


def test_auc_of_perfect_ranker_is_one():
    """Any constant added to a metric breaks its known fixed points."""
    y = np.array([0, 0, 1, 1])
    assert roc_auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(np.array([5.0]), np.array([100.0]))
    assert lo[0] < 0.05 < hi[0]
    # must stay inside [0,1] even at the boundary, unlike the normal approximation
    lo0, hi0 = wilson_interval(np.array([0.0]), np.array([14.0]))
    assert lo0[0] == pytest.approx(0.0, abs=1e-9) and hi0[0] < 1.0


def test_rate_by_code_shares_sum_to_one(uci):
    assert rate_by_code(uci, "EDUCATION")["share"].sum() == pytest.approx(1.0)
