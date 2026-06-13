"""
Almgren-Chriss (2000) optimal execution.

You have to liquidate X units over a horizon T. Trade fast and you pay market impact;
trade slow and you're exposed to price risk (volatility) the whole time. Almgren-Chriss
makes that trade-off precise: minimise  E[cost] + lambda * Var[cost], and the optimal
holding path is a closed-form exponential — the more risk-averse you are (higher
lambda), the faster you get out.

Impact model (linear): temporary cost  eta * (trade rate)  plus a half-spread epsilon;
permanent impact  gamma  per unit traded; price diffuses with volatility sigma.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class AlmgrenChriss:
    X: float            # units to liquidate
    T: float            # horizon (same time unit as sigma, e.g. days)
    N: int              # number of trading intervals
    sigma: float        # price volatility per unit time
    eta: float          # temporary-impact slope (per unit trade rate)
    gamma: float = 0.0  # permanent-impact slope
    epsilon: float = 0.0  # fixed temporary cost (half bid-ask spread)
    lam: float = 1e-6   # risk aversion

    @property
    def tau(self) -> float:
        return self.T / self.N

    @property
    def eta_tilde(self) -> float:
        return self.eta - 0.5 * self.gamma * self.tau

    def kappa(self) -> float:
        """Decay rate of the optimal trajectory. Bigger lambda/sigma -> bigger kappa
        -> faster liquidation. Discrete form: cosh(kappa*tau) = 1 + 0.5 k2 tau^2."""
        k2 = self.lam * self.sigma ** 2 / self.eta_tilde
        return float(np.arccosh(1.0 + 0.5 * k2 * self.tau ** 2) / self.tau)

    def trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Times and holdings x_k (units still to sell). Exponential decay to 0."""
        k = self.kappa()
        t = np.arange(self.N + 1) * self.tau
        x = self.X * np.sinh(k * (self.T - t)) / np.sinh(k * self.T)
        x[0], x[-1] = self.X, 0.0
        return t, x

    def trade_list(self) -> np.ndarray:
        """Units sold in each interval (the schedule you actually send)."""
        _, x = self.trajectory()
        return -np.diff(x)

    def expected_cost(self) -> float:
        n = self.trade_list()
        permanent = 0.5 * self.gamma * self.X ** 2
        temporary = self.epsilon * np.sum(np.abs(n)) + (self.eta_tilde / self.tau) * np.sum(n ** 2)
        return float(permanent + temporary)

    def variance(self) -> float:
        _, x = self.trajectory()
        return float(self.sigma ** 2 * self.tau * np.sum(x[1:] ** 2))

    def twap(self) -> tuple[float, float]:
        """Cost and variance of the naive equal-slice (TWAP) schedule, for comparison."""
        twap = replace(self, lam=0.0)
        n = np.full(self.N, self.X / self.N)
        permanent = 0.5 * self.gamma * self.X ** 2
        temporary = self.epsilon * np.sum(np.abs(n)) + (twap.eta_tilde / self.tau) * np.sum(n ** 2)
        x = np.concatenate([[self.X], self.X - np.cumsum(n)])
        var = self.sigma ** 2 * self.tau * np.sum(x[1:] ** 2)
        return float(permanent + temporary), float(var)
