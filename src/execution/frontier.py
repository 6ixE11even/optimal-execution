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
            "half_life": float(np.log(2) / ac.kappa()),
        })
    return pd.DataFrame(rows)
