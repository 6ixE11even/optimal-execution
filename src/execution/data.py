"""
Calibrate execution parameters from real market data (Deribit public API).

Volatility (sigma) and average daily volume (ADV) are pulled from a real instrument so
the impact/risk inputs aren't made up. The temporary-impact slope eta is then scaled by
ADV (a common rule of thumb: impact grows as you take a larger share of daily volume).
"""
from __future__ import annotations

import time

import numpy as np
import requests

DERIBIT_CHART = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"


def calibrate_from_deribit(instrument: str = "BTC-PERPETUAL", days: int = 180) -> dict:
    """Returns {price, sigma (daily, in price units), adv} from real OHLCV."""
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    resp = requests.get(DERIBIT_CHART, params={"instrument_name": instrument, "resolution": "1D",
                                               "start_timestamp": start, "end_timestamp": end}, timeout=30)
    resp.raise_for_status()
    result = resp.json()["result"]
    close = np.asarray(result["close"], dtype=float)
    volume = np.asarray(result["volume"], dtype=float)

    daily_log_ret = np.diff(np.log(close))
    sigma_price = float(close[-1] * daily_log_ret.std())   # $ stdev per day
    return {"price": float(close[-1]), "sigma": sigma_price, "adv": float(volume.mean())}
