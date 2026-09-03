"""Trajectory (optimal vs TWAP) and the cost/risk efficient frontier."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from execution.model import AlmgrenChriss


def plot_trajectory(ac: AlmgrenChriss, out_path: str | Path) -> None:
    t, x = ac.trajectory()
    twap = ac.X * (1 - t / ac.T)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, x / ac.X * 100, color="#3b6ea5", lw=2, label=f"Almgren-Chriss (half-life {np.log(2)/ac.kappa():.2f})")
    ax.plot(t, twap / ac.X * 100, color="#c0392b", lw=1.4, ls="--", label="TWAP (linear)")
    ax.set_title("Optimal liquidation trajectory", fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("% of position remaining")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, out_path)


def plot_frontier(frontier, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frontier["stdev"], frontier["expected_cost"], color="#2c7a4b", lw=2, marker="o", ms=3)
    ax.set_title("Efficient frontier of execution", fontweight="bold")
    ax.set_xlabel("risk  (std of execution cost)")
    ax.set_ylabel("expected cost")
    ax.grid(alpha=0.25)
    _save(fig, out_path)


def _save(fig, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(history, frames, path):
    """Training objective per epoch, with the out-of-sample level the policies land on."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(history) + 1), history, "o-", color="#1f77b4", label="REINFORCE (train)")
    test = frames["test"]
    for name, colour, style in (("Almgren-Chriss", "#d62728", "--"),
                                ("TWAP", "#7f7f7f", ":"),
                                ("REINFORCE", "#1f77b4", "-.")):
        ax.axhline(test.query("policy == @name").objective_bps.mean(),
                   color=colour, ls=style, lw=1.2, label=f"{name} (test)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("objective: shortfall + risk penalty (bps)")
    ax.set_title("Learning curve against the two benchmarks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_holdings(test_frame, path):
    """Average inventory path of each policy, out of sample."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, colour in (("Almgren-Chriss", "#d62728"), ("TWAP", "#7f7f7f"),
                         ("REINFORCE", "#1f77b4")):
        paths = np.vstack(test_frame.query("policy == @name").holdings.to_list())
        mean = paths.mean(axis=0)
        ax.plot(np.arange(1, len(mean) + 1), mean, label=name, color=colour)
        ax.fill_between(np.arange(1, len(mean) + 1),
                        np.percentile(paths, 25, axis=0), np.percentile(paths, 75, axis=0),
                        color=colour, alpha=0.12, lw=0)
    ax.set_xlabel("bar")
    ax.set_ylabel("inventory remaining (fraction of order)")
    ax.set_title("Average liquidation path, out of sample (band: interquartile)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_objective_gap(test_frame, path):
    """Per-episode objective differences. The point is how wide these are."""
    ac = test_frame.query("policy == 'Almgren-Chriss'").objective_bps.to_numpy()
    tw = test_frame.query("policy == 'TWAP'").objective_bps.to_numpy()
    rl = test_frame.query("policy == 'REINFORCE'").objective_bps.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(*np.percentile(np.concatenate([ac - tw, rl - tw]), [0.5, 99.5]), 60)
    ax.hist(ac - tw, bins=bins, alpha=0.55, label="Almgren-Chriss - TWAP", color="#d62728")
    ax.hist(rl - tw, bins=bins, alpha=0.55, label="REINFORCE - TWAP", color="#1f77b4")
    ax.axvline(0, color="k", lw=0.8)
    for d, colour in ((ac - tw, "#d62728"), (rl - tw, "#1f77b4")):
        ax.axvline(d.mean(), color=colour, ls="--", lw=1.4)
    ax.set_xlabel("objective difference vs TWAP (bps, per episode)")
    ax.set_ylabel("episodes")
    ax.set_title("Both edges are small next to the spread of a single execution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
