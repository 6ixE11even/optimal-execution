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
