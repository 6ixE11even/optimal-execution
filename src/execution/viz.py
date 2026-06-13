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
