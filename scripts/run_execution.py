"""
Compute the optimal liquidation schedule and the efficient frontier.

    python scripts/run_execution.py                    # real Deribit calibration
    python scripts/run_execution.py --participation 0.05
    python scripts/run_execution.py --offline          # illustrative params, no network
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
    parser.add_argument("--offline", action="store_true",
                        help="skip the network and use illustrative parameters")
    parser.add_argument("--instrument", default="BTC-PERPETUAL")
    parser.add_argument("--participation", type=float, default=0.10,
                        help="order size as a fraction of ADV (default: 0.10)")
    parser.add_argument("--half-life", type=float, default=1.0,
                        help="urgency, as the trajectory half-life in days (default: 1.0)")
    args = parser.parse_args()

    size_note = f"{args.participation:.0%} of ADV"
    if args.offline:
        sigma, eta, epsilon, price = 2.0, 2.5e-6, 0.0625, 100.0
        X = 1_000_000
        size_note = "fixed size, no ADV reference"
        print("offline: illustrative parameters, not calibrated to any instrument")
    else:
        from execution.data import calibrate_from_deribit
        cal = calibrate_from_deribit(args.instrument)
        sigma, price = cal["sigma"], cal["price"]
        # Temporary impact must come out in price units. eta = 1/ADV has units of
        # 1/(units per day) and produced impact of two cents a unit on a $77,000
        # instrument - 0.0 bps on a $41M order, which is not a cost model.
        # eta = sigma/ADV says trading one full ADV in a day costs about one daily
        # volatility of slippage, the standard linear rule of thumb, and is
        # dimensionally consistent: [$/day] / [units/day] = $ per unit of rate.
        eta = sigma / cal["adv"]
        epsilon = cal["half_spread"]
        # Size the order off the book we just measured. 1,000,000 units against a
        # 5,377-unit ADV is 186 days of volume, and linear impact says nothing there.
        X = args.participation * cal["adv"]
        print(f"{args.instrument}: price {price:,.0f}  sigma {sigma:,.0f} $/day  "
              f"ADV {cal['adv']:,.0f}  half-spread {epsilon:.2f}")

    # lambda is unit-dependent; pick it from the half-life the user actually wants.
    probe = AlmgrenChriss(X=X, T=5, N=20, sigma=sigma, eta=eta, gamma=eta / 10,
                          epsilon=epsilon, lam=1e-12)
    ac = AlmgrenChriss(X=X, T=5, N=20, sigma=sigma, eta=eta, gamma=eta / 10,
                       epsilon=epsilon, lam=probe.lam_for_half_life(args.half_life))
    cost, var = ac.expected_cost(), ac.variance()
    twap_cost, twap_var = ac.twap()
    notional = ac.X * price
    print(f"\nLiquidate {ac.X:,.0f} units ({size_note}, "
          f"${notional:,.0f} notional) over T={ac.T} days in {ac.N} slices")
    print(f"{'':<17}{'cost ($)':>14}{'  (bps)':>9}{'risk std ($)':>16}{'half-life':>11}")
    print(f"  Almgren-Chriss {cost:>14,.0f}{1e4 * cost / notional:>9.1f}"
          f"{np.sqrt(var):>16,.0f}{np.log(2) / ac.kappa():>11.2f}")
    print(f"  TWAP           {twap_cost:>14,.0f}{1e4 * twap_cost / notional:>9.1f}"
          f"{np.sqrt(twap_var):>16,.0f}{'—':>11}")

    # Sweep urgency from "a tenth of the window" to "the whole window", which is the
    # readable range; a raw lambda grid is only meaningful for one instrument's units.
    half_lives = np.geomspace(ac.T / 20, ac.T * 2, 25)
    frontier = efficient_frontier(ac, np.array([probe.lam_for_half_life(h) for h in half_lives]))
    out = ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    frontier.to_csv(out / "frontier.csv", index=False)
    plot_trajectory(ac, out / "figures" / "trajectory.png")
    plot_frontier(frontier, out / "figures" / "efficient_frontier.png")
    print(f"\nwrote -> {out}/ (frontier.csv, figures/)")


if __name__ == "__main__":
    main()
