"""
Calibrate execution parameters from real market data (Deribit public API).

Volatility (sigma), average daily volume (ADV) and the live touch spread are pulled
from a real instrument so the impact/risk inputs aren't made up. The temporary-impact
slope eta is then scaled by ADV (a common rule of thumb: impact grows as you take a
larger share of daily volume), and epsilon is half the quoted spread.

The order size has to come from ADV too. Calibrating sigma and eta to a real book and
then liquidating a hardcoded 1,000,000 units gives an order 186x the instrument's daily
volume, where a linear temporary-impact model does not describe anything. Size is set
as a participation rate instead.
"""
from __future__ import annotations

import time

import numpy as np
import requests

DERIBIT_CHART = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
DERIBIT_TICKER = "https://www.deribit.com/api/v2/public/ticker"


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

    # Half the live touch spread is the fixed cost of crossing, which is what
    # epsilon means in Almgren-Chriss. It was hardcoded at 0.0625 - an eighth,
    # i.e. a US equity tick from before decimalisation.
    tick = requests.get(DERIBIT_TICKER, params={"instrument_name": instrument}, timeout=30)
    tick.raise_for_status()
    t = tick.json()["result"]
    half_spread = 0.5 * (float(t["best_ask_price"]) - float(t["best_bid_price"]))

    return {"price": float(close[-1]), "sigma": sigma_price,
            "adv": float(volume.mean()), "half_spread": half_spread}
