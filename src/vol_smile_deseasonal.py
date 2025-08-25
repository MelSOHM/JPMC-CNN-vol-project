import numpy as np
import pandas as pd
from typing import Dict, Tuple, Hashable

_RSC = 1.4826  # robust scale constant to transform MAD into a std-like scale
_EPS = 1e-12

def _tod_keys(idx: pd.DatetimeIndex, bucket: str = "minute"):
    """Return time-of-day keys for grouping: ('hour','minute') if bucket='minute', else 'hour'."""
    if bucket == "minute":
        return list(zip(idx.hour, idx.minute))
    elif bucket == "hour":
        return idx.hour
    else:
        raise ValueError("bucket must be 'minute' or 'hour'")

def _robust_median_scale(x: np.ndarray) -> Tuple[float, float]:
    """Compute (median, robust_scale) where robust_scale = 1.4826 * MAD; never returns < EPS."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (0.0, 1.0)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(_RSC * mad, _EPS)
    return med, scale

def fit_robust_tod_stats(vol: pd.Series, bucket: str = "minute") -> Dict[Hashable, Tuple[float, float]]:
    """
    Fit robust time-of-day statistics on TRAIN ONLY (no look-ahead):
      For each time-of-day bucket b, compute:
        med_b = median(vol | bucket=b)
        sca_b = 1.4826 * MAD(vol | bucket=b)
    Returns a dict bucket -> (med_b, sca_b), plus a '_global' fallback.
    """
    if not isinstance(vol.index, pd.DatetimeIndex):
        raise TypeError("vol.index must be a DatetimeIndex")
    keys = _tod_keys(vol.index, bucket=bucket)
    df = pd.DataFrame({"vol": vol.values, "key": keys})
    stats: Dict[Hashable, Tuple[float, float]] = {}
    for k, sub in df.groupby("key", sort=False):
        med, sca = _robust_median_scale(sub["vol"].to_numpy())
        stats[k] = (med, sca)
    # global fallback used when a bucket is unseen in train
    stats["_global"] = _robust_median_scale(vol.to_numpy())
    return stats

def apply_robust_tod_zscore(vol: pd.Series, stats: Dict[Hashable, Tuple[float, float]], bucket: str = "minute") -> pd.Series:
    """
    Apply robust z-score de-seasonalization using pre-fitted time-of-day stats:
      z_t = (vol_t - med_b) / sca_b   where b = bucket(t)
    Uses '_global' fallback if a bucket was unseen during fitting.
    """
    if not isinstance(vol.index, pd.DatetimeIndex):
        raise TypeError("vol.index must be a DatetimeIndex")
    keys = _tod_keys(vol.index, bucket=bucket)
    g_med, g_sca = stats.get("_global", (0.0, 1.0))
    out = pd.Series(index=vol.index, dtype=float, name=vol.name + "_zds")
    for ts, v, k in zip(vol.index, vol.values, keys):
        med, sca = stats.get(k, (g_med, g_sca))
        out.loc[ts] = (v - med) / (sca if sca > _EPS else _EPS) if np.isfinite(v) else np.nan
    return out

def deseason_robust_expanding(vol: pd.Series, bucket: str = "minute") -> pd.Series:
    """
    Strict ex-ante variant (slower): for each time-of-day bucket, compute
    expanding median and expanding MAD up to t-1, then z_t = (x_t - med_{t-1}) / (1.4826 * MAD_{t-1}).
    This avoids any look-ahead at the point level, but is computationally heavier.
    """
    if not isinstance(vol.index, pd.DatetimeIndex):
        raise TypeError("vol.index must be a DatetimeIndex")
    keys = _tod_keys(vol.index, bucket=bucket)
    df = pd.DataFrame({"vol": vol, "key": keys})
    z = pd.Series(index=vol.index, dtype=float, name=vol.name + "_zds")

    def _mad_exp(arr: np.ndarray) -> float:
        m = np.median(arr)
        return np.median(np.abs(arr - m))

    for k, sub in df.groupby("key", sort=False):
        v = sub["vol"]
        med = v.expanding().median().shift(1)
        mad = v.expanding().apply(_mad_exp, raw=True).shift(1)
        sca = (_RSC * mad).clip(lower=_EPS)
        z.loc[sub.index] = (v - med) / sca
    return z
