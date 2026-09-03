"""
Real intraday bars from Deribit, paginated and cached.

The public chart endpoint returns at most 5,001 bars per call whatever window you
ask for, and it does not tell you that it truncated - it just hands back a shorter
series with `status: ok`. A single request for two years of five-minute data
therefore looks like it worked and silently covers seventeen days. This walks
backwards in chunks and stitches the pieces together.

Bars are cached as csv.gz next to the repo so a rerun is free and so the numbers in
the README correspond to a file someone can open.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CHART = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
MAX_BARS = 5_000                      # the endpoint's hard cap, minus one for safety
MINUTES = {"1": 1, "3": 3, "5": 5, "10": 10, "15": 15, "30": 30, "60": 60,
           "120": 120, "180": 180, "360": 360, "720": 720, "1D": 1440}
CACHE = Path(__file__).resolve().parents[2] / "data" / "bars"


def _chunk(instrument: str, resolution: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    resp = requests.get(CHART, params={"instrument_name": instrument, "resolution": resolution,
                                       "start_timestamp": start_ms, "end_timestamp": end_ms},
                        timeout=60)
    resp.raise_for_status()
    result = resp.json()["result"]
    if result.get("status") == "no_data":
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    return pd.DataFrame({
        "ts": pd.to_datetime(result["ticks"], unit="ms", utc=True),
        "open": result["open"], "high": result["high"], "low": result["low"],
        "close": result["close"], "volume": result["volume"],
    })


def fetch_bars(instrument: str = "BTC-PERPETUAL", resolution: str = "5",
               days: int = 720, refresh: bool = False) -> pd.DataFrame:
    """OHLCV bars over the last `days`, stitched across as many calls as it takes."""
    if resolution not in MINUTES:
        raise ValueError(f"resolution {resolution!r} not one of {sorted(MINUTES)}")
    path = CACHE / f"{instrument}_{resolution}m_{days}d.csv.gz"
    if path.exists() and not refresh:
        bars = pd.read_csv(path, parse_dates=["ts"])
        return bars

    step_ms = MINUTES[resolution] * 60_000
    end = int(time.time() * 1000)
    floor = end - days * 86_400_000
    frames, cursor = [], end
    while cursor > floor:
        start = max(floor, cursor - MAX_BARS * step_ms)
        chunk = _chunk(instrument, resolution, start, cursor)
        if chunk.empty:
            break
        frames.append(chunk)
        oldest = int(chunk["ts"].iloc[0].timestamp() * 1000)
        if oldest >= cursor:                  # no progress; the series starts here
            break
        cursor = oldest - step_ms
        time.sleep(0.1)                       # Deribit's published public limit is 20/s

    bars = (pd.concat(frames)
              .drop_duplicates(subset="ts")
              .sort_values("ts")
              .reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(path, index=False)
    return bars


def episode_windows(bars: pd.DataFrame, n_steps: int) -> np.ndarray:
    """Start indices of non-overlapping, gap-free windows of `n_steps` bars.

    Non-overlapping matters: overlapping windows share most of their price path, and
    an agent scored on them is being tested on data it trained on. Gap-free matters
    because Deribit has maintenance halts, and a window that straddles one contains a
    price jump no execution algorithm could have traded through.
    """
    ts = bars["ts"].to_numpy()
    spacing = np.diff(ts).astype("timedelta64[s]").astype(float)
    modal = np.median(spacing)
    starts, i = [], 0
    while i + n_steps < len(bars):
        if np.all(spacing[i:i + n_steps] <= modal * 1.5):
            starts.append(i)
            i += n_steps
        else:
            i += 1
    return np.asarray(starts, dtype=int)
