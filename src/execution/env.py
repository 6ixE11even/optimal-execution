"""
A liquidation episode played out on a real price path.

Every episode is a contiguous window of real Deribit bars. The agent starts holding
X units and must be flat by the last bar; what it earns is the cash it actually
collects, and what it is scored on is implementation shortfall against the arrival
price. The impact model is the same one Almgren-Chriss assumes - linear temporary
impact, linear permanent impact, a half-spread paid on every fill - so the closed
form and a learned policy are graded by identical arithmetic and any difference
between them is the schedule, not the cost function.

Nothing in the state is dated after the bar the agent is standing on. Volatility,
average volume and the spread estimate all come from the window *before* the
episode starts, which is the difference between a backtest and a look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from execution.model import AlmgrenChriss

N_FEATURES = 5


def corwin_schultz_spread(high: np.ndarray, low: np.ndarray) -> float:
    """Proportional bid-ask spread estimated from high/low ranges (Corwin-Schultz 2012).

    Deribit's chart endpoint carries no quote history, so the alternative to an
    estimator is a hardcoded number of basis points. This one needs only the highs
    and lows already in the bars: a single bar's range reflects volatility plus the
    spread, two bars pooled reflect twice the volatility plus one spread, and the
    difference identifies the spread. Negative estimates happen in quiet windows and
    are clipped to zero, as the paper prescribes.
    """
    high, low = np.asarray(high, float), np.asarray(low, float)
    ok = (high > 0) & (low > 0)
    if ok.sum() < 3:
        return 0.0
    log_hl = np.log(high[ok] / low[ok]) ** 2
    beta = log_hl[:-1] + log_hl[1:]
    h2 = np.maximum(high[ok][:-1], high[ok][1:])
    l2 = np.minimum(low[ok][:-1], low[ok][1:])
    gamma = np.log(h2 / l2) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return float(np.maximum(spread, 0.0).mean())


@dataclass
class EpisodeSpec:
    """Everything about one episode that is known before the first trade."""
    price: np.ndarray      # close of each bar in the window, length N+1
    X: float               # units to liquidate
    sigma: float           # price stdev per bar, from the lookback window
    eta: float             # temporary impact slope, $ per unit of trade rate
    gamma: float           # permanent impact slope
    epsilon: float         # half-spread in price units
    lam: float             # risk aversion, the same one Almgren-Chriss is given
    tau: float = 1.0       # bar length in the time unit sigma is quoted in

    @property
    def N(self) -> int:
        return len(self.price) - 1


def build_episodes(bars, starts, n_steps: int, participation: float,
                   lookback: int, half_life: float) -> list[EpisodeSpec]:
    """Turn raw bars into episodes, calibrating each one on the bars that precede it.

    Urgency is given as a half-life in bars, not as a risk aversion. lambda carries the
    instrument's price and volume units, so one number cannot mean the same thing in two
    episodes whose price differs by a factor of three; a half-life does, and the lambda
    that produces it is solved for per episode. At lambda small enough to be effectively
    risk-neutral the Almgren-Chriss path collapses to TWAP and there is nothing for an
    agent to learn that TWAP does not already know.
    """
    close = bars["close"].to_numpy(float)
    high, low = bars["high"].to_numpy(float), bars["low"].to_numpy(float)
    volume = bars["volume"].to_numpy(float)

    episodes = []
    for s in starts:
        if s < lookback:
            continue                          # not enough history to calibrate on
        past = slice(s - lookback, s)
        ret = np.diff(np.log(close[past]))
        if not np.isfinite(ret).all() or ret.std() == 0:
            continue
        # Scale by the last bar that closed *before* the window opens. Using the
        # episode's own first close is nearly harmless and completely indefensible:
        # it is a price from inside the test set leaking into the parameters the
        # benchmark schedules are built from.
        sigma = float(close[s - 1] * ret.std())         # price units per bar
        adv = float(volume[past].mean()) * n_steps     # volume over an episode's length
        if adv <= 0:
            continue
        spread = corwin_schultz_spread(high[past], low[past])
        eta = sigma / adv                     # one episode's volume costs one sigma
        spec = EpisodeSpec(
            price=close[s:s + n_steps + 1],
            X=participation * adv,
            sigma=sigma,
            eta=eta,
            gamma=eta / 10.0,                 # permanent impact a tenth of temporary
            epsilon=0.5 * spread * close[s - 1],
            lam=0.0,
        )
        spec.lam = lam_for_half_life(spec, half_life)
        episodes.append(spec)
    return episodes


def lam_for_half_life(spec: EpisodeSpec, half_life: float) -> float:
    """The risk aversion whose Almgren-Chriss trajectory decays with this half-life."""
    return AlmgrenChriss(X=spec.X, T=spec.N * spec.tau, N=spec.N, sigma=spec.sigma,
                         eta=spec.eta, gamma=spec.gamma, epsilon=spec.epsilon,
                         lam=1e-12).lam_for_half_life(half_life)


def ac_fractions(spec: EpisodeSpec) -> np.ndarray:
    """The closed-form trajectory, expressed as a fraction of remaining inventory.

    `run_schedule` speaks fractions so that a deterministic schedule and a state-
    dependent policy go through one code path. Converting is exact: if the optimal
    holding falls from x_k to x_{k+1}, the fraction sold is 1 - x_{k+1}/x_k.
    """
    ac = AlmgrenChriss(X=spec.X, T=spec.N * spec.tau, N=spec.N, sigma=spec.sigma,
                       eta=spec.eta, gamma=spec.gamma, epsilon=spec.epsilon, lam=spec.lam)
    _, x = ac.trajectory()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(x[:-1] > 0, 1.0 - x[1:] / x[:-1], 1.0)
    return np.clip(frac, 0.0, 1.0)


def twap_fractions(n_steps: int) -> np.ndarray:
    """Equal slices, as fractions of what is left: 1/N, 1/(N-1), ... , 1."""
    return np.array([1.0 / (n_steps - k) for k in range(n_steps)])


def features(spec: EpisodeSpec, k: int, inventory: float) -> np.ndarray:
    """What the agent sees at step k. Everything is unit-free so one policy transfers
    across episodes whose price levels differ by a factor of three."""
    p = spec.price
    elapsed = max(k, 1)
    drift = (p[k] - p[0]) / (spec.sigma * np.sqrt(elapsed))
    last = (p[k] - p[k - 1]) / spec.sigma if k > 0 else 0.0
    return np.array([
        1.0 - k / spec.N,                     # time remaining
        inventory / spec.X,                   # inventory remaining
        np.clip(drift, -4, 4),                # how far the path has run against us
        np.clip(last, -4, 4),                 # the most recent move
        (spec.N - k) * inventory / (spec.X * spec.N),   # interaction: slack vs urgency
    ], dtype=float)


def run_schedule(spec: EpisodeSpec, fractions: np.ndarray) -> dict:
    """Execute a sequence of sell fractions and return the realised economics.

    `fractions[k]` is the share of *remaining* inventory sold in bar k. The last bar
    is forced to 1.0 whatever the policy asked for: an execution algorithm that ends
    the day holding stock has not solved the problem it was given.
    """
    x, cash, cost_steps, held = spec.X, 0.0, [], []
    permanent = 0.0
    for k in range(spec.N):
        f = 1.0 if k == spec.N - 1 else float(np.clip(fractions[k], 0.0, 1.0))
        n = x * f
        rate = n / spec.tau
        # A trade walks the book down as it goes, so it fills at the midpoint of its
        # own permanent impact, not at the far end of it. Charging the full gamma*n
        # to the trade that caused it overstates the cost of front-loading by
        # 0.5*gamma*sum(n_k^2) and hands a rigged comparison to the slower schedule.
        # This half-interval is exactly what Almgren-Chriss folds into eta_tilde.
        exec_price = (spec.price[k] - permanent - 0.5 * spec.gamma * n
                      - spec.eta * rate - spec.epsilon * (n > 0))
        permanent += spec.gamma * n
        cash += n * exec_price
        x -= n
        held.append(x)
        cost_steps.append(n * (spec.price[0] - exec_price))
    shortfall = spec.X * spec.price[0] - cash
    held = np.asarray(held)
    risk = spec.lam * spec.sigma ** 2 * spec.tau * float(np.sum(held ** 2))
    return {"shortfall": float(shortfall), "risk_penalty": float(risk),
            "objective": float(shortfall + risk), "cost_steps": np.asarray(cost_steps),
            "holdings": held, "notional": float(spec.X * spec.price[0]),
            "residual": float(x)}
