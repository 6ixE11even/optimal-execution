"""The efficient frontier of execution: each risk-aversion gives one (cost, variance)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from execution.model import AlmgrenChriss


def efficient_frontier(base: AlmgrenChriss, lambdas) -> pd.DataFrame:
    rows = []
    for lam in lambdas:
        ac = replace(base, lam=lam)
        rows.append({
            "lambda": lam,
            "expected_cost": ac.expected_cost(),
            "variance": ac.variance(),
            "stdev": np.sqrt(ac.variance()),
            # kappa -> 0 in the risk-neutral limit, where the schedule is a straight
            # line and "half-life" has no meaning. Report inf rather than dividing by
            # zero and emitting a RuntimeWarning into the middle of a results table.
            "half_life": float(np.log(2) / k) if (k := ac.kappa()) > 0 else float("inf"),
        })
    return pd.DataFrame(rows)
