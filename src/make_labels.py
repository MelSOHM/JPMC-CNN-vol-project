import pandas as pd
from pandas.tseries.offsets import BusinessDay
import numpy as np

def rolling_median_ex_ante(x: pd.Series, window: int) -> pd.Series:
    """
    Rolling median only based on historical data (excluding today).
    """
    return x.shift(1).rolling(window=window, min_periods=window).median()

def make_labels_in_sample(vol: pd.Series,
                median_window: int = 26,   # kept for API compatibility (unused)
                drop_na: bool = True) -> pd.DataFrame:
    med = rolling_median_ex_ante(vol, window=median_window-1).rename("median_hist")
    print("median window", median_window-1)
    # future vol at first timestamp >= t + horizon_days (in hour)
    vfut = vol.rename(f"vol_tplus_0D")

    y = (vfut > med).astype("float").rename(f"y_h0")
    out = pd.concat({"vol": vol, "median_hist": med, f"y_h0": y}, axis=1)
    if drop_na:
        out = out.dropna()
    return out

def lookahead_by_days(s: pd.Series, days: int) -> pd.Series:
    """
    Return a series aligned on s.index whose value at time t is s at the first
    timestamp >= (t + days). Works with tz-aware DatetimeIndex and irregular grids.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("lookahead_by_bdays expects a DatetimeIndex")
    idx = s.index
    target = idx + BusinessDay(days)
    pos = idx.get_indexer(target, method="bfill")
    out = pd.Series(np.nan, index=idx, name=f"{s.name}_tplus_{days}B")
    ok = pos != -1
    if ok.any():
        vals = s.to_numpy()
        out.iloc[ok] = vals[pos[ok]]
    return out

def make_labels(vol: pd.Series,
                horizon_days: int = 1,
                median_window: int = 100,
                drop_na: bool = True) -> pd.DataFrame:
    """
    Build ex-ante binary labels using a *calendar-day* horizon:
    y_t = 1{ vol_{first ts >= t + horizon_days} > median_hist_t },
    where median_hist_t is the rolling historical median up to t-1.
    Notes:
    - horizon measured in days (calendar).
    - if no future timestamp exists, label is NaN and will be dropped.
    """
    # historical median (no look-ahead at t)
    med = rolling_median_ex_ante(vol, window=median_window).rename("median_hist")
    
    # future vol at first timestamp >= t + horizon_days
    vfut = lookahead_by_days(vol, horizon_days).rename(f"vol_tplus_{horizon_days}D")

    y = (vfut > med).astype("float").rename(f"y_h{horizon_days}")
    out = pd.concat({"vol": vol, "median_hist": med, f"y_h{horizon_days}": y}, axis=1)
    if drop_na:
        out = out.dropna()
    return out

def _infer_step_timedelta(idx: pd.DatetimeIndex) -> pd.Timedelta:
    try:
        f = pd.infer_freq(idx)
        if f:
            return pd.tseries.frequencies.to_offset(f).delta
    except Exception:
        pass
    if len(idx) < 2:
        return pd.Timedelta(0)
    diffs_ns = np.diff(idx.view("i8"))
    return pd.Timedelta(int(np.median(diffs_ns)), unit="ns")

def make_labels_overnight(
    ohlcv: pd.DataFrame,
    horizon_days: int = 1,
    median_window: int = 100,
    drop_na: bool = True,
    *,
    tz_local: str = "America/New_York",
    open_time: str = "09:30",
    close_time: str = "16:00",
    bar_label: str = "end",         # "end" si timestamp = fin de bougie, "start" sinon
    use_business_days: bool = True,  # True => BDay (week-ends exclus), False => jours calendaires
) -> pd.DataFrame:
    """Return format identical to base make_labels(): ['vol','median_hist',f'y_h{horizon_days}'] on the same intraday index."""
    req = {"open", "close"}
    if not req.issubset(ohlcv.columns):
        raise ValueError(f"Missing columns: {req - set(ohlcv.columns)}")
    if not isinstance(ohlcv.index, pd.DatetimeIndex) or ohlcv.index.tz is None:
        raise TypeError("ohlcv index must be a tz-aware UTC DatetimeIndex.")

    step = _infer_step_timedelta(ohlcv.index)

    # Local timestamps to detect per-day open/close bars
    idx_loc   = ohlcv.index.tz_convert(tz_local)
    days_loc  = idx_loc.normalize()
    open_loc  = pd.to_datetime(days_loc.strftime("%Y-%m-%d") + " " + open_time).tz_localize(tz_local)
    close_loc = pd.to_datetime(days_loc.strftime("%Y-%m-%d") + " " + close_time).tz_localize(tz_local)
    ts_open_expect  = open_loc + step if bar_label == "end" else open_loc
    ts_close_expect = close_loc       if bar_label == "end" else close_loc - step

    mask_open  = (idx_loc == ts_open_expect)
    mask_close = (idx_loc == ts_close_expect)

    open_series  = ohlcv.loc[mask_open,  "open"].copy()
    close_series = ohlcv.loc[mask_close, "close"].copy()

    # If missing sessions, return empty frame with correct columns
    colnames = ["vol", "median_hist", f"y_h{horizon_days}"]
    if close_series.empty or open_series.empty:
        return pd.DataFrame(index=ohlcv.index, columns=colnames).dropna()

    # For each local EOD day t, pick next day's open at +h days
    eod_days_loc = close_series.index.tz_convert(tz_local).normalize()
    if use_business_days:
        tgt_days_loc = eod_days_loc + BusinessDay(horizon_days)
    else:
        tgt_days_loc = eod_days_loc + pd.Timedelta(days=horizon_days)

    tgt_open_loc = pd.to_datetime(tgt_days_loc.strftime("%Y-%m-%d") + " " + open_time).tz_localize(tz_local)
    tgt_open_ts  = tgt_open_loc + step if bar_label == "end" else tgt_open_loc
    tgt_open_utc = tgt_open_ts.tz_convert("UTC")

    # Map EOD -> next open price
    open_fwd = open_series.reindex(tgt_open_utc)
    open_fwd.index = close_series.index  # align on EOD index

    # Overnight per day (EOD index)
    ov_day = (np.log(open_fwd) - np.log(close_series)).abs()
    ov_day.name = "vol"
    med_day = ov_day.shift(1).rolling(median_window, min_periods=median_window).median()
    med_day.name = "median_hist"

    # Broadcast to *all intraday bars* of day t (keep same number of windows)
    day2ov  = pd.Series(ov_day.values, index=eod_days_loc)   # key: local day -> ov value
    day2med = pd.Series(med_day.values, index=eod_days_loc)

    base_days_loc = ohlcv.index.tz_convert(tz_local).normalize()
    # map() on an Index returns ndarray → wrap back into Series with the original intraday index
    vol_full = pd.Series(base_days_loc.map(day2ov.to_dict()), index=ohlcv.index, name="vol")
    med_full = pd.Series(base_days_loc.map(day2med.to_dict()), index=ohlcv.index, name="median_hist")

    yname  = f"y_h{horizon_days}"
    y_full = pd.Series((vol_full.values > med_full.values).astype(float), index=ohlcv.index, name=yname)

    out = pd.concat([vol_full, med_full, y_full], axis=1)
    return out.dropna() if drop_na else out

def make_labels_ternary(
    vol: pd.Series,
    horizon_days: int = 1,
    median_window: int = 100,
    *,
    band_value: float = 0.3,          # intensité de la bande
    band_mode: str = "pct",           # "abs" | "pct" | "mad"
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Ternary labels with a dead-band around the rolling ex-ante median.
    Output columns: ["vol", "median_hist", f"y_h{horizon_days}"], where y ∈ {-1,0,1}.
    Thresholds:
        abs : [m - band, m + band]
        pct : [m*(1-band), m*(1+band)]
        mad : [m - band*MAD, m + band*MAD] with ex-ante rolling MAD
    """
    med = rolling_median_ex_ante(vol, window=median_window).rename("median_hist")

    if band_mode == "abs":
        lo, hi = med - band_value, med + band_value
    elif band_mode == "pct":
        lo, hi = med * (1 - band_value), med * (1 + band_value)
    elif band_mode == "mad":
        r = vol.shift(1).rolling(median_window, min_periods=median_window)
        mad = r.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        lo, hi = med - band_value * mad, med + band_value * mad
    else:
        raise ValueError("band_mode must be one of {'abs','pct','mad'}")

    vfut = lookahead_by_days(vol, horizon_days).rename(f"vol_tplus_{horizon_days}D")
    up = (vfut > hi).astype(int)
    dn = -(vfut < lo).astype(int)
    y = (up + dn).rename(f"y_h{horizon_days}")  # in {-1,0,1}

    out = pd.concat({"vol": vol, "median_hist": med, y.name: y}, axis=1)
    return out.dropna() if drop_na else out