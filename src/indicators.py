# indicators.py
# Lightweight, vectorized technical indicators (no look-ahead).
from __future__ import annotations
import numpy as np
import pandas as pd

_EPS = 1e-12

def sma(x: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Simple moving average."""
    return x.rolling(window=window, min_periods=min_periods).mean()

def ema(x: pd.Series, span: int, adjust: bool = False, min_periods: int = 1) -> pd.Series:
    """Exponential moving average."""
    return x.ewm(span=span, adjust=adjust, min_periods=min_periods).mean()

def rolling_std(x: pd.Series, window: int, min_periods: int = 1, ddof: int = 0) -> pd.Series:
    """Rolling standard deviation (population by default, ddof=0)."""
    return x.rolling(window=window, min_periods=min_periods).std(ddof=ddof)

def bollinger_bands(x: pd.Series, window: int = 20, nstd: float = 2.0,
                    min_periods: int = 1) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands on a generic series:
      mid = SMA(x, window)
      upper = mid + nstd * rolling_std(x, window)
      lower = mid - nstd * rolling_std(x, window)
    Returned series are aligned to x.index and may start with NaNs if not enough history.
    """
    mid = sma(x, window=window, min_periods=min_periods)
    sd = rolling_std(x, window=window, min_periods=min_periods, ddof=0)
    upper = mid + nstd * sd
    lower = mid - nstd * sd
    mid.name = f"bb_mid_{window}"
    upper.name = f"bb_up_{window}_{nstd}"
    lower.name = f"bb_lo_{window}_{nstd}"
    return mid, upper, lower

def rsi_wilder(x: pd.Series, window: int = 14) -> pd.Series:
    """
    Wilder's RSI on a generic series (works on prices or any 1D signal like volatility):
      1) delta = diff(x)
      2) gains = max(delta, 0), losses = max(-delta, 0)
      3) Wilder smoothing with alpha = 1/window
      4) RSI = 100 - 100/(1 + avg_gain/avg_loss)
    Output in [0, 100], NaN for the first 'window' points.
    """
    delta = x.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder's smoothing is equivalent to an EMA with alpha=1/window (adjust=False)
    avg_gain = up.ewm(alpha=1.0/window, adjust=False, min_periods=window).mean()
    avg_loss = down.ewm(alpha=1.0/window, adjust=False, min_periods=window).mean()

    rs = avg_gain / (avg_loss + _EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = f"rsi_{window}"
    return rsi

def zscore(x: pd.Series, window: int = 20, min_periods: int = 5) -> pd.Series:
    """
    Rolling z-score: (x - mean)/std using rolling window.
    Useful as an additional normalized overlay (e.g., on volatility).
    """
    mean = x.rolling(window=window, min_periods=min_periods).mean()
    std = x.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (x - mean) / (std.replace(0, np.nan))
    z.name = f"z_{window}"
    return z
