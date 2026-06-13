"""
Compute the optimal liquidation schedule and the efficient frontier.

    python scripts/run_execution.py              # example parameters (offline)
    python scripts/run_execution.py --calibrate  # pull real sigma/ADV from Deribit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution.frontier import efficient_frontier   # noqa: E402
from execution.model import AlmgrenChriss            # noqa: E402
from execution.viz import plot_frontier, plot_trajectory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Almgren-Chriss optimal execution.")
    parser.add_argument("--calibrate", action="store_true", help="pull real sigma/ADV from Deribit")
    parser.add_argument("--instrument", default="BTC-PERPETUAL")
    args = parser.parse_args()

    sigma, eta = 2.0, 2.5e-6
    if args.calibrate:
        from execution.data import calibrate_from_deribit
        cal = calibrate_from_deribit(args.instrument)
        sigma = cal["sigma"]
        eta = 1.0 / cal["adv"]   # impact ~ 1/ADV: deeper books are cheaper to trade
        print(f"calibrated from {args.instrument}: price={cal['price']:.1f} sigma={sigma:.3f} ADV={cal['adv']:.0f}")

    ac = AlmgrenChriss(X=1_000_000, T=5, N=20, sigma=sigma, eta=eta, gamma=eta / 10,
                       epsilon=0.0625, lam=2e-6)
    cost, var = ac.expected_cost(), ac.variance()
    twap_cost, twap_var = ac.twap()
    print(f"\nLiquidate {ac.X:,.0f} units over T={ac.T} in {ac.N} slices")
    print(f"  Almgren-Chriss : cost {cost:>12,.0f}   risk(std) {np.sqrt(var):>12,.0f}   half-life {np.log(2)/ac.kappa():.2f}")
    print(f"  TWAP           : cost {twap_cost:>12,.0f}   risk(std) {np.sqrt(twap_var):>12,.0f}")

    frontier = efficient_frontier(ac, np.logspace(-8, -4, 25))
    out = ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    frontier.to_csv(out / "frontier.csv", index=False)
    plot_trajectory(ac, out / "figures" / "trajectory.png")
    plot_frontier(frontier, out / "figures" / "efficient_frontier.png")
    print(f"\nwrote -> {out}/ (frontier.csv, figures/)")


if __name__ == "__main__":
    main()
