"""
Train a policy-gradient execution agent on real bars and grade it against the closed form.

    uv run python scripts/train_rl_execution.py
    uv run python scripts/train_rl_execution.py --epochs 10 --participation 0.05

Episodes are non-overlapping windows of real Deribit five-minute bars. The split is
chronological, never random: a random split would put the afternoon of a day in
training and its morning in test, and BTC's own autocorrelation would then flatter
the agent. Everything reported below is on windows that begin after the last window
the agent trained on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution.bars import episode_windows, fetch_bars          # noqa: E402
from execution.env import (ac_fractions, build_episodes,        # noqa: E402
                           run_schedule, twap_fractions)
from execution.rl import GaussianPolicy, rollout, train         # noqa: E402
from execution.viz import (plot_holdings, plot_learning_curve,  # noqa: E402
                           plot_objective_gap)


def evaluate(episodes, fractions_fn, name: str) -> pd.DataFrame:
    rows = []
    for i, spec in enumerate(episodes):
        out = run_schedule(spec, fractions_fn(spec))
        bps = 1e4 / out["notional"]
        rows.append({"policy": name, "episode": i,
                     "shortfall_bps": out["shortfall"] * bps,
                     "risk_bps": out["risk_penalty"] * bps,
                     "objective_bps": out["objective"] * bps,
                     "holdings": out["holdings"] / spec.X})
    return pd.DataFrame(rows)


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str = "objective_bps") -> dict:
    d = a[col].to_numpy() - b[col].to_numpy()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean_diff": float(d.mean()), "se": float(se),
            "t": float(d.mean() / se) if se else float("nan"),
            "win_rate": float((d < 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="BTC-PERPETUAL")
    ap.add_argument("--days", type=int, default=720)
    ap.add_argument("--resolution", default="5")
    ap.add_argument("--steps", type=int, default=36, help="bars per episode")
    ap.add_argument("--lookback", type=int, default=288, help="bars used to calibrate")
    ap.add_argument("--participation", type=float, default=0.10)
    ap.add_argument("--half-life", type=float, default=12.0, help="urgency, in bars")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-control-variate", action="store_true",
                    help="train on raw shortfall instead of the TWAP-differenced reward")
    args = ap.parse_args()

    bars = fetch_bars(args.instrument, args.resolution, args.days)
    starts = episode_windows(bars, args.steps)
    episodes = build_episodes(bars, starts, args.steps, args.participation,
                              args.lookback, args.half_life)
    cut = int(len(episodes) * args.train_frac)
    tr_eps, te_eps = episodes[:cut], episodes[cut:]
    print(f"{args.instrument} {args.resolution}m: {len(bars):,} bars "
          f"{bars.ts.min():%Y-%m-%d} -> {bars.ts.max():%Y-%m-%d}")
    print(f"{len(episodes):,} non-overlapping {args.steps}-bar episodes  "
          f"({len(tr_eps):,} train / {len(te_eps):,} test)")
    print(f"order size {args.participation:.0%} of the window's volume, "
          f"half-life {args.half_life:.0f} bars\n")

    policy = GaussianPolicy(seed=args.seed)
    history = train(policy, tr_eps, epochs=args.epochs, lr=args.lr, seed=args.seed,
                    control=not args.no_control_variate)
    print("training objective by epoch (bps):",
          "  ".join(f"{h:.3f}" for h in history))

    rng = np.random.default_rng(args.seed + 1)

    def rl_fractions(spec):
        return rollout(policy, spec, rng, greedy=True)["fractions"]

    frames = {}
    for split, eps in (("train", tr_eps), ("test", te_eps)):
        frames[split] = pd.concat([
            evaluate(eps, ac_fractions, "Almgren-Chriss"),
            evaluate(eps, lambda s: twap_fractions(s.N), "TWAP"),
            evaluate(eps, rl_fractions, "REINFORCE"),
        ], ignore_index=True).assign(split=split)
    results = pd.concat(frames.values(), ignore_index=True)

    print(f"\n{'':<16}{'shortfall':>11}{'risk':>8}{'objective':>11}   (bps, mean over episodes)")
    for split in ("train", "test"):
        print(f"  --- {split} ({len(frames[split]) // 3:,} episodes)")
        for name in ("Almgren-Chriss", "TWAP", "REINFORCE"):
            g = frames[split].query("policy == @name")
            print(f"  {name:<16}{g.shortfall_bps.mean():>10.3f}"
                  f"{g.risk_bps.mean():>8.3f}{g.objective_bps.mean():>11.3f}")

    te = frames["test"]
    tests = [("Almgren-Chriss", "TWAP"), ("REINFORCE", "Almgren-Chriss"), ("REINFORCE", "TWAP")]
    print(f"\n  paired, out of sample ({len(te) // 3:,} episodes)")
    print(f"  {'':<32}{'diff':>8}{'se':>7}{'t':>7}{'wins':>8}")
    stats = []
    for a, b in tests:
        s = paired(te.query("policy == @a").reset_index(), te.query("policy == @b").reset_index())
        stats.append({"a": a, "b": b, **s})
        print(f"  {a + ' - ' + b:<32}{s['mean_diff']:>8.3f}{s['se']:>7.3f}"
              f"{s['t']:>7.2f}{s['win_rate']:>7.1%}")

    out = ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    results.drop(columns="holdings").to_csv(out / "rl_execution.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "rl_execution_tests.csv", index=False)
    plot_learning_curve(history, frames, out / "figures" / "rl_learning_curve.png")
    plot_holdings(frames["test"], out / "figures" / "rl_holdings.png")
    plot_objective_gap(frames["test"], out / "figures" / "rl_objective_gap.png")
    print(f"\nwrote -> {out}/ (rl_execution.csv, rl_execution_tests.csv, figures/)")


if __name__ == "__main__":
    main()
