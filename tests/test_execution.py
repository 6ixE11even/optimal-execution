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


def test_half_life_round_trips_through_lambda():
    """lambda is unit-dependent; a half-life is not. Solving one for the other has
    to land back where it started, or urgency is not actually specifiable."""
    ac = AlmgrenChriss(X=1_000, T=5, N=20, sigma=1_494.0, eta=1_494.0 / 5_377,
                       gamma=(1_494.0 / 5_377) / 10, epsilon=2.0, lam=1e-12)
    for target in (0.25, 1.0, 2.5, 10.0):
        lam = ac.lam_for_half_life(target)
        tuned = replace(ac, lam=lam)
        assert abs(np.log(2) / tuned.kappa() - target) < 1e-6


def test_half_life_must_be_positive():
    ac = AlmgrenChriss(X=1_000, T=5, N=20, sigma=2.0, eta=2.5e-6,
                       gamma=2.5e-7, epsilon=0.0625, lam=1e-9)
    for bad in (0.0, -1.0):
        try:
            ac.lam_for_half_life(bad)
        except ValueError as e:
            assert "half_life must be positive" in str(e)
        else:
            raise AssertionError(f"expected ValueError for half_life={bad}")


# --- the RL execution agent -------------------------------------------------

import pytest

import pandas as pd  # noqa: E402

from execution.bars import episode_windows  # noqa: E402
from execution.env import (EpisodeSpec, ac_fractions, build_episodes,  # noqa: E402
                           corwin_schultz_spread, features, run_schedule,
                           twap_fractions)
from execution.rl import GaussianPolicy, control_variate, rollout  # noqa: E402


def _flat_spec(n=12, price=100.0, **kw):
    args = dict(price=np.full(n + 1, price), X=1_000.0, sigma=1.0, eta=0.0,
                gamma=0.0, epsilon=0.0, lam=0.0)
    args.update(kw)
    return EpisodeSpec(**args)


def test_corwin_schultz_recovers_a_spread_it_was_not_told_about():
    """Widen a random walk's highs and lows by a known spread; get the spread back."""
    rng = np.random.default_rng(11)
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, 4_000)))
    true_spread = 0.0020                                   # 20 bps, proportional
    high = mid * (1 + abs(rng.normal(0, 0.003, mid.size)) + true_spread / 2)
    low = mid * (1 - abs(rng.normal(0, 0.003, mid.size)) - true_spread / 2)
    est = corwin_schultz_spread(high, low)
    assert 0.5 * true_spread < est < 2.0 * true_spread, est


def test_a_schedule_always_finishes_flat_whatever_the_policy_asked_for():
    """An execution algorithm that ends the horizon still holding has not finished."""
    spec = _flat_spec()
    for fractions in (np.zeros(spec.N), np.full(spec.N, 0.01), np.full(spec.N, 0.9)):
        assert run_schedule(spec, fractions)["residual"] == pytest.approx(0.0)


def test_no_impact_and_no_spread_on_a_flat_price_costs_nothing():
    spec = _flat_spec()
    out = run_schedule(spec, twap_fractions(spec.N))
    assert out["shortfall"] == pytest.approx(0.0, abs=1e-8)


def test_ac_fractions_reproduce_the_closed_form_holdings():
    """The fraction-of-remaining encoding has to be a faithful re-parameterisation."""
    spec = _flat_spec(n=20, sigma=2.0, eta=1e-4, gamma=1e-5, epsilon=0.01, lam=1e-6)
    ac = AlmgrenChriss(X=spec.X, T=spec.N, N=spec.N, sigma=spec.sigma, eta=spec.eta,
                       gamma=spec.gamma, epsilon=spec.epsilon, lam=spec.lam)
    _, x = ac.trajectory()
    walked = run_schedule(spec, ac_fractions(spec))["holdings"]
    assert np.allclose(walked, x[1:], rtol=1e-9, atol=1e-9)


def test_calibration_cannot_see_past_the_start_of_its_own_episode():
    """Change every bar from the episode's first onwards; the calibration must not move."""
    n = 40
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 300)))
    bars = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=300, freq="5min"),
                         "open": close, "high": close * 1.001, "low": close * 0.999,
                         "close": close, "volume": np.full(300, 10.0)})
    starts = np.array([200])
    before = build_episodes(bars, starts, n, 0.1, lookback=100, half_life=10.0)[0]

    tampered = bars.copy()
    tampered.loc[200:, ["open", "high", "low", "close"]] *= 3.0
    tampered.loc[200:, "volume"] *= 50.0
    after = build_episodes(tampered, starts, n, 0.1, lookback=100, half_life=10.0)[0]

    for field in ("X", "sigma", "eta", "gamma", "epsilon", "lam"):
        assert getattr(before, field) == pytest.approx(getattr(after, field)), field


def test_episode_windows_refuses_to_straddle_a_data_gap():
    ts = pd.date_range("2025-01-01", periods=60, freq="5min").to_list()
    ts = ts[:30] + [t + pd.Timedelta(hours=6) for t in ts[30:]]   # a maintenance halt
    bars = pd.DataFrame({"ts": ts, "close": np.arange(60.0)})
    for start in episode_windows(bars, 10):
        assert not (start < 30 <= start + 10), f"window at {start} crosses the halt"


def test_policy_gradient_matches_finite_differences():
    """The backward pass is written by hand, so it is checked by hand."""
    policy = GaussianPolicy(hidden=6, seed=5)
    s = np.array([0.6, 0.4, 0.3, -0.2, 0.24])
    _, z, mu, cache = policy.act(s, np.random.default_rng(0))
    analytic = policy.grad_log_prob(s, z, mu, cache)

    def log_prob():
        m, _ = policy.forward(s)
        std = float(np.exp(policy.log_std[0]))
        return -0.5 * ((z - m) / std) ** 2 - np.log(std)

    eps = 1e-6
    for name, tensor in (("W1", policy.W1), ("W2", policy.W2), ("W3", policy.W3),
                         ("b2", policy.b2), ("log_std", policy.log_std)):
        flat, grad = tensor.reshape(-1), analytic[name].reshape(-1)
        for i in range(min(5, flat.size)):
            keep = flat[i]
            flat[i] = keep + eps
            up = log_prob()
            flat[i] = keep - eps
            down = log_prob()
            flat[i] = keep
            assert grad[i] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-7), name


def test_the_control_variate_does_not_depend_on_the_agent():
    """It is only a valid control variate if the action cannot move it."""
    spec = _flat_spec(n=15, price=100.0, sigma=1.0, eta=1e-4, epsilon=0.02, lam=1e-6)
    spec.price = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.001, spec.N + 1)))
    first = control_variate(spec)
    for seed in (0, 1, 2):
        rollout(GaussianPolicy(seed=seed), spec, np.random.default_rng(seed))
        assert np.allclose(control_variate(spec), first)


def test_features_are_scale_free():
    """Doubling the price level must not move the state the policy sees."""
    spec = _flat_spec(n=10, price=100.0, sigma=1.0)
    spec.price = np.linspace(100.0, 110.0, spec.N + 1)
    doubled = _flat_spec(n=10, price=200.0, sigma=2.0, X=spec.X)
    doubled.price = spec.price * 2
    for k in (0, 3, 9):
        assert np.allclose(features(spec, k, spec.X * 0.4),
                           features(doubled, k, doubled.X * 0.4))
