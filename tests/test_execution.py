"""The optimal trajectory must liquidate the full position, beat TWAP on the
risk-adjusted objective, and speed up as risk-aversion rises."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.model import AlmgrenChriss

BASE = AlmgrenChriss(X=1_000_000, T=5, N=20, sigma=2.0, eta=2.5e-6, gamma=2.5e-7, epsilon=0.0625, lam=2e-6)


def test_trade_list_liquidates_full_position():
    assert abs(BASE.trade_list().sum() - BASE.X) < 1e-3


def test_trajectory_is_monotone_decreasing():
    _, x = BASE.trajectory()
    assert np.all(np.diff(x) <= 1e-6)


def test_beats_twap_on_objective():
    twap_cost, twap_var = BASE.twap()
    ac_obj = BASE.expected_cost() + BASE.lam * BASE.variance()
    twap_obj = twap_cost + BASE.lam * twap_var
    assert ac_obj <= twap_obj + 1.0           # AC is optimal for E + lambda*Var
    assert BASE.variance() < twap_var          # and carries less price risk


def test_more_risk_averse_liquidates_faster():
    slow = replace(BASE, lam=1e-7).kappa()
    fast = replace(BASE, lam=1e-5).kappa()
    assert fast > slow                          # higher kappa = faster decay


def test_risk_neutral_limit_is_twap_not_nan():
    """lam -> 0 used to return an all-NaN path: sinh(0)/sinh(0).

    With no risk aversion there is nothing to weigh against impact, so the optimal
    schedule is the straight line. Check the path is finite and linear, and that its
    cost matches the TWAP cost computed independently.
    """
    ac = AlmgrenChriss(X=1_000_000, T=5, N=20, sigma=2.0, eta=2.5e-6,
                       gamma=2.5e-7, epsilon=0.0625, lam=0.0)
    _, x = ac.trajectory()
    assert np.all(np.isfinite(x))
    slices = -np.diff(x)
    assert np.allclose(slices, slices[0]), "risk-neutral schedule should be equal-sliced"
    twap_cost, twap_var = ac.twap()
    assert abs(ac.expected_cost() - twap_cost) / twap_cost < 1e-9
    assert abs(ac.variance() - twap_var) / twap_var < 1e-9


def test_trajectory_is_continuous_as_lambda_vanishes():
    """No discontinuity at the cutoff where the closed form hands over to the line."""
    base = dict(X=1_000_000, T=5, N=20, sigma=2.0, eta=2.5e-6, gamma=2.5e-7, epsilon=0.0)
    costs = [AlmgrenChriss(**base, lam=lam).expected_cost()
             for lam in (1e-14, 1e-13, 1e-12, 1e-11)]
    assert np.all(np.isfinite(costs))
    assert max(costs) / min(costs) - 1 < 1e-6


def test_degenerate_impact_is_rejected_with_a_useful_message():
    """0.5*gamma*tau >= eta breaks convexity; arccosh then silently returned NaN."""
    import pytest
    ac = AlmgrenChriss(X=1_000, T=5, N=20, sigma=2.0, eta=1e-8, gamma=1.0, lam=1e-6)
    with pytest.raises(ValueError, match="eta_tilde"):
        ac.expected_cost()
