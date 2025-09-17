#!/usr/bin/env python3
# build_dataset.py
# Dataset builder: OHLCV -> GK vol -> labels (t+1/t+2)
# Author: Mel & Daisy (JPMC project)
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple
import re

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay
from pandas.tseries.offsets import Hour

from PIL import Image
import matplotlib.pyplot as plt

from . import vol_smile_deseasonal as vol_smile
from . import indicators
from tqdm.auto import tqdm
from .image_generations import to_heatmap_image, to_recurrence_image, to_timeseries_image, to_gaf_image, compute_global_indicators
# from .qa_data import count_missing_minutesx_fixed_window

# ---------------------------
# 1) Loading & Prep
# ---------------------------

# --- YAML config loader/merger -----------------------------------------------
from types import SimpleNamespace

try:
    import yaml
except ImportError as e:
    yaml = None 

def _resolve_config_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    here = Path(__file__).parent
    return (here / p).resolve()

from pathlib import Path

def load_yaml_config(path: str | Path) -> dict:
    """
    Robust YAML resolver:
    - If absolute path -> use it.
    - Else try, in order: CWD/path, <script_dir>/path, <script_dir>/../path.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for --config. Install with: pip install pyyaml")

    p = Path(path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        here = Path(__file__).parent
        candidates += [Path.cwd() / p, here / p, here.parent / p]

    tried = []
    for cand in candidates:
        tried.append(str(cand))
        if cand.exists():
            with open(cand, "r") as f:
                return yaml.safe_load(f) or {}

    raise FileNotFoundError(f"Config file not found. Tried: {tried}")

def merge_args_with_config(args) -> SimpleNamespace:
    """
    Merge CLI args with YAML config (if provided).
    CLI overrides config; config provides defaults when CLI is empty.
    Returns a SimpleNamespace with all fields the script expects.
    """
    cfg = {}
    if getattr(args, "config", None):
        cfg = load_yaml_config(args.config)

    def get(cfg_path, default=None):
        d = cfg
        for key in cfg_path.split("."):
            if not isinstance(d, dict) or key not in d:
                return default
            d = d[key]
        return d

    # Base (CLI wins over YAML)
    csv = args.csv or get("data.csv")
    out = args.out or get("output.dir")
    batch_out = get("output.batch_dir", None)
    symbol = args.symbol or get("data.symbol")
    tz_utc = get("data.tz_utc", True)
    time_col = getattr(args, "time_col", None) or get("data.time_col", None)
    
    # resample
    rs_rule    = get("data.resample.rule", None) or getattr(args, "rs_rule", None)
    rs_label   = getattr(args, "rs_label", None)   or get("data.resample.label", "left")
    rs_closed  = getattr(args, "rs_closed", None)  or get("data.resample.closed", "left")
    rs_dropna  = get("data.resample.drop_na", True) if getattr(args, "rs_dropna", None) is None else getattr(args, "rs_dropna")

    train_end = args.train_end or get("splits.train_end")
    val_end   = args.val_end or get("splits.val_end")

    horizons = args.horizons or get("labels.horizons", [1, 2])
    median_window = args.median_window if args.median_window is not None else get("labels.median_window", 100)
    label_mode = get("labels.mode", "standard")

    deseason_mode = args.deseason_mode or get("deseasonalization.mode", "none")
    deseason_bucket = args.deseason_bucket or get("deseasonalization.bucket", "minute")

    image_encoder = args.image_encoder or get("images.encoder", "heatmap")
    image_windows = args.image_windows or get("images.windows", [128])
    image_step = args.image_step if args.image_step is not None else get("images.step", 1)
    ts_ma_window = args.ts_ma_window if args.ts_ma_window is not None else get("images.ts_ma_window", 60)

    img_w = args.img_w if args.img_w is not None else get("images.size.width", 256)
    img_h = args.img_h if args.img_h is not None else get("images.size.height", 256)
    img_dpi = args.img_dpi if args.img_dpi is not None else get("images.size.dpi", 100)
    
    # Aligment start of day
    align_enabled   = get("images.alignment.enabled", False)
    align_time_str  = get("images.alignment.anchor_time", "14:00")
    align_tz        = get("images.alignment.anchor_tz", "UTC")

    heatmap_mode = get("images.heatmap.mode", "colormap")
    heatmap_cmap = get("images.heatmap.cmap", "viridis")
    heatmap_vmin = get("images.heatmap.vmin", None)
    heatmap_vmax = get("images.heatmap.vmax", None)

    # --- Day separator (ts_vol) ---
    day_sep_enabled = get("images.ts_overlays.day_separators.enabled", False)
    day_sep_label   = get("images.ts_overlays.day_separators.label", True)
    day_sep_color   = get("images.ts_overlays.day_separators.color", "#888888")
    day_sep_alpha   = get("images.ts_overlays.day_separators.alpha", 0.35)
    day_sep_lw      = get("images.ts_overlays.day_separators.linewidth", 0.8)
    day_label_kind  = get("images.ts_overlays.day_separators.label_kind", "number")
    
    # --- overlays (ts_vol) ---
    show_ma_top  = get("images.ts_overlays.top.show_ma", True)
    bb_enabled   = get("images.ts_overlays.top.show_bbands", False)
    bb_window    = get("images.ts_overlays.top.bb_window", 20)
    bb_nstd      = get("images.ts_overlays.top.bb_nstd", 2.0)

    bottom_panel = get("images.ts_overlays.bottom.panel", "volume")  # volume|rsi|none
    rsi_window   = get("images.ts_overlays.bottom.rsi_window", 14)
    rsi_source  = get("images.ts_overlays.bottom.rsi_source", "vol")  # 'vol' or 'close'

    # --- style (shared) ---
    fg = get("images.ts_overlays.style.fg", "white")
    bg = get("images.ts_overlays.style.bg", "black")
    
    # --- Reccurence Plot ---
    rp_series        = get("images.recurrence.series", "vol")
    rp_normalize     = get("images.recurrence.normalize", "zscore")
    rp_metric        = get("images.recurrence.metric", "euclidean")
    rp_epsilon_mode  = get("images.recurrence.epsilon_mode", "quantile")
    rp_epsilon_q     = get("images.recurrence.epsilon_q", 0.10)
    rp_epsilon_value = get("images.recurrence.epsilon_value", None)
    rp_binarize      = get("images.recurrence.binarize", True)
    rp_cmap          = get("images.recurrence.cmap", "gray")
    rp_invert        = get("images.recurrence.invert", True)
    
    # --- Gamian Angular field ---
    gaf_mode      = get("images.gaf.mode", "gasf")
    gaf_normalize = get("images.gaf.normalize", "minmax")
    gaf_cmap      = get("images.gaf.cmap", "viridis")
    gaf_invert    = get("images.gaf.invert", False)

    embargo_steps = args.embargo_steps if getattr(args, "embargo_steps", None) is not None else get("evaluation.embargo_steps", 0)
    no_overlap = get("evaluation.no_overlap", False)

    # Apply no_overlap policy (if requested) by making step == max(window)
    if no_overlap and image_windows:
        image_step = max(image_windows)

    # Basic validation
    missing = []
    if not csv:    missing.append("--csv or data.csv")
    if not out:    missing.append("--out or output.dir")
    if not symbol: missing.append("--symbol or data.symbol")
    if missing:
        raise ValueError("Missing required config: " + ", ".join(missing))

    return SimpleNamespace(
        csv=csv, out=out, batch_out=batch_out, symbol=symbol, tz_utc=tz_utc, time_col=time_col,
        rs_rule=rs_rule, rs_label=rs_label, rs_closed=rs_closed, rs_dropna=rs_dropna,
        train_end=train_end, val_end=val_end,
        horizons=horizons, median_window=median_window, label_mode=label_mode,
        deseason_mode=deseason_mode, deseason_bucket=deseason_bucket,
        image_encoder=image_encoder, image_windows=image_windows, image_step=image_step,
        align_enabled=align_enabled, align_time_str=align_time_str, align_tz=align_tz,
        ts_ma_window=ts_ma_window, img_w=img_w, img_h=img_h, img_dpi=img_dpi,
        heatmap_mode=heatmap_mode, heatmap_cmap=heatmap_cmap,
        heatmap_vmin=heatmap_vmin, heatmap_vmax=heatmap_vmax,
        day_sep_enabled=day_sep_enabled,
        day_sep_label=day_sep_label,
        day_sep_color=day_sep_color,
        day_sep_alpha=day_sep_alpha,
        day_sep_lw=day_sep_lw,
        day_label_kind=day_label_kind,
        embargo_steps=embargo_steps, show_ma_top=show_ma_top,
        bb_enabled=bb_enabled, bb_window=bb_window, bb_nstd=bb_nstd,
        bottom_panel=bottom_panel, rsi_window=rsi_window, rsi_source=rsi_source,
        fg=fg, bg=bg, rp_series=rp_series, rp_normalize=rp_normalize, rp_metric=rp_metric,
        rp_epsilon_mode=rp_epsilon_mode, rp_epsilon_q=rp_epsilon_q,
        rp_epsilon_value=rp_epsilon_value, rp_binarize=rp_binarize,
        rp_cmap=rp_cmap, rp_invert=rp_invert,gaf_mode=gaf_mode, gaf_normalize=gaf_normalize,
        gaf_cmap=gaf_cmap, gaf_invert=gaf_invert,
    )
    
def ts_to_filename(ts) -> str:
    """Safe file name from a timestamp or any index-like key."""
    try:
        t = pd.to_datetime(ts, utc=True)
        return t.strftime("%Y%m%dT%H%M%S%fZ")
    except Exception:
        s = str(ts)
        return re.sub(r"[^0-9A-Za-z_-]+", "-", s)

def load_ohlcv(csv_path: Path,
               tz_utc: bool = True,
               time_col: str | None = None,
               *,
               rs_rule: str | None = None,
               rs_label: str = "left",
               rs_closed: str = "left",
               rs_dropna: bool = True) -> pd.DataFrame:
    """
    Expect a CSV with columns: <time>, open, high, low, close[, volume].
    Sets a tz-aware DatetimeIndex. Optionally resamples to a coarser bar.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # choose time_col (auto-detect if not provided)
    if time_col is None:
        for c in ["timestamp", "ts_event", "datetime", "time", "date"]:
            if c in df.columns:
                time_col = c
                break
    if time_col is None or time_col not in df.columns:
        raise ValueError(f"Cannot find time column. Provide data.time_col. Columns: {list(df.columns)}")

    df[time_col] = pd.to_datetime(df[time_col], utc=tz_utc, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    req = {"open", "high", "low", "close"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns {miss}. Present: {list(df.columns)}")

    if rs_rule:
        df = resample_ohlcv(df, rs_rule, label=rs_label, closed=rs_closed, drop_na=rs_dropna)
    return df

def resample_ohlcv(df: pd.DataFrame,
                   rule: str,
                   *,
                   label: str = "left",
                   closed: str = "left",
                   drop_na: bool = True) -> pd.DataFrame:
    """
    Resample an OHLC(V) DataFrame to a coarser bar (e.g., 30min, 1h).
    - open: first, high: max, low: min, close: last, volume: sum (if present)
    - Keeps tz-aware DatetimeIndex and sorts the index.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low":  "min",
        "close":"last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"

    # Only aggregate columns we actually have
    cols = [c for c in agg.keys() if c in df.columns]
    out = (
        df[cols]
        .resample(rule, label=label, closed=closed)
        .agg({c: agg[c] for c in cols})
        .sort_index()
    )

    if drop_na:
        out = out.dropna(subset=["open", "high", "low", "close"])
    return out

# ------------------------------------------
# 2) Garman–Klass Volatility (interval-based)
# ------------------------------------------

def garman_klass_sigma(df: pd.DataFrame,
                       use_sigma: bool = True) -> pd.Series:
    """
    Compute GK Var by interval and the vol (sqrt) if use_sigma
    GK variance:
        0.5 * [ln(H/L)]^2 - (2 ln 2 - 1) * [ln(C/O)]^2
    """
    H, L, C, O = df["high"], df["low"], df["close"], df["open"]
    lnHL2 = np.log(H / L) ** 2
    lnCO2 = np.log(C / O) ** 2
    var_gk = 0.5 * lnHL2 - (2 * math.log(2) - 1) * lnCO2
    var_gk = var_gk.clip(lower=0)  # robustesse num.
    if use_sigma:
        return np.sqrt(var_gk).rename("sigma_gk")
    return var_gk.rename("var_gk")

# -----------------------------------------------
# 3) Rolling Median and ex-ante t+h labels
# -----------------------------------------------

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

def make_labels_overnight(ohlcv: pd.DataFrame,
                         median_window: int = 100,
                         exclude_weekends: bool = True,
                         market_tz: str = "America/New_York") -> pd.DataFrame:
    """
    Build ex-ante binary labels for overnight volatility prediction:
      y_t = 1{ overnight_vol_{t+1_open} > median_hist_t },
    where overnight_vol = |log(Open_{t+1}) - log(Close_t)| and median_hist_t 
    is the rolling historical median of overnight_vol up to t-1.
    
    Args:
        ohlcv: DataFrame with OHLC columns and DatetimeIndex
        median_window: window for rolling median (in sessions, not bars)
        exclude_weekends: if True, exclude Friday->Monday gaps
        market_tz: timezone for market hours (default: America/New_York)
    
    Returns:
        DataFrame with columns: close_ts, overnight_vol, median_hist, y_h1
        Indexed by the close timestamps (day t)
    """
    if not all(col in ohlcv.columns for col in ["open", "high", "low", "close"]):
        raise ValueError("OHLCV must contain 'open', 'high', 'low', 'close' columns")
    
    # Convert to market timezone for reliable session identification
    df_market = ohlcv.copy()
    if df_market.index.tz is None:
        df_market.index = df_market.index.tz_localize("UTC")
    df_market.index = df_market.index.tz_convert(market_tz)
    
    # Find session close (16:00) and next session open (09:30)
    session_closes = []
    session_opens = []
    # Group by date and find 16:00 close and 09:30 open
    for date, day_data in df_market.groupby(df_market.index.date):
        # Find 16:00 close (last bar at or before 16:00)
        close_candidates = day_data[day_data.index.time <= pd.Timestamp("16:00").time()]
        if not close_candidates.empty:
            close_ts = close_candidates.index[-1]  # last bar <= 16:00
            session_closes.append((close_ts, day_data.loc[close_ts, "close"]))
        
        # Find 09:30 open (first bar at or after 09:30)
        open_candidates = day_data[day_data.index.time >= pd.Timestamp("09:30").time()]
        if not open_candidates.empty:
            open_ts = open_candidates.index[0]  # first bar >= 09:30
            session_opens.append((open_ts, day_data.loc[open_ts, "open"]))
    
    if not session_closes or not session_opens:
        raise ValueError("Could not find valid session closes/opens")
    
    # Create DataFrames for closes and opens
    closes_df = pd.DataFrame(session_closes, columns=["timestamp", "close_price"])
    opens_df = pd.DataFrame(session_opens, columns=["timestamp", "open_price"])
    closes_df.set_index("timestamp", inplace=True)
    opens_df.set_index("timestamp", inplace=True)
    
    # Map each close to the next business day's open
    overnight_pairs = []
    for close_ts, close_price in closes_df.itertuples():
        close_date = close_ts.date()
        
        # Find next business day's open
        next_opens = opens_df[opens_df.index.date > close_date]
        if next_opens.empty:
            continue
            
        next_open_ts = next_opens.index[0]
        next_open_date = next_open_ts.date()
        next_open_price = next_opens.loc[next_open_ts, "open_price"]
        
        # Exclude weekend gaps (Friday -> Monday) if requested
        if exclude_weekends:
            days_gap = (next_open_date - close_date).days
            if days_gap > 1:  # More than 1 day gap (weekend/holiday)
                continue
        
        # Calculate overnight squared return: (log(open) - log(close))^2
        overnight_vol = (np.log(next_open_price) - np.log(close_price)) ** 2
        
        overnight_pairs.append({
            "close_ts": close_ts,
            "next_open_ts": next_open_ts,
            "close_price": close_price,
            "next_open_price": next_open_price,
            "overnight_vol": overnight_vol
        })
    
    if not overnight_pairs:
        raise ValueError("No valid overnight pairs found")
    
    # Create DataFrame with overnight volatilities
    overnight_df = pd.DataFrame(overnight_pairs)
    overnight_df.set_index("close_ts", inplace=True)
    overnight_df = overnight_df.sort_index()
    
    # Calculate rolling median of historical overnight volatilities (no look-ahead)
    overnight_vol_series = overnight_df["overnight_vol"]
    median_hist = overnight_vol_series.shift(1).rolling(window=median_window, min_periods=median_window).median()
    
    # Create binary labels: 1 if overnight_vol > median_hist
    y = (overnight_vol_series > median_hist).astype("float").rename("y_h1")
    
    # Combine results
    result = pd.concat({
        "overnight_vol": overnight_vol_series,
        "median_hist": median_hist,
        "y_h1": y
    }, axis=1)
    
    # Add metadata
    result["next_open_ts"] = overnight_df["next_open_ts"]
    result["close_price"] = overnight_df["close_price"]
    result["next_open_price"] = overnight_df["next_open_price"]
    
    # Drop rows with NaN (insufficient history for median)
    result = result.dropna()
    
    return result

# -------------------------------------------------
# 4) (train/val/test) temporal split 
# -------------------------------------------------

def ensure_datetime_index(df: pd.DataFrame, time_col: str | None = None, utc: bool = True) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.sort_index()
        if utc and out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        return out
    # if the time col exists 
    if time_col and time_col in df.columns:
        idx = pd.to_datetime(df[time_col], utc=utc, errors="coerce")
        out = df.set_index(idx).sort_index()
        return out
    # otherwise we convert the index into datetime
    try:
        idx = pd.to_datetime(df.index, utc=utc, errors="coerce")
        out = df.copy()
        out.index = idx
        out = out.sort_index()
        return out
    except Exception as e:
        raise TypeError("A DatetimeIndex or a valid time_col is required for time_split.") from e
    
def realign_to_day_anchor(df: pd.DataFrame,
                          anchor_time: str = "14:00",
                          anchor_tz: str = "UTC") -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """
    Slice the DataFrame so that the *first row* is the first timestamp >=
    the day 'anchor_time' in 'anchor_tz'.
    - Keeps original tz/index; we only use a tz-converted *view* to find the cut index.
    - If no anchor found (e.g., tail-short), returns (df, None).

    Example:
      If df starts at 13:30 and anchor=14:00, the first kept row will be the first bar >= 14:00
      of that same day (or the next day if today's 14:00 is already past).
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df, None

    idx = df.index
    # convert to anchor tz (view only)
    idx_local = (idx.tz_convert(anchor_tz) if idx.tz is not None
                 else idx.tz_localize("UTC").tz_convert(anchor_tz))

    # parse HH:MM
    try:
        hh, mm = map(int, anchor_time.split(":")[:2])
    except Exception as e:
        raise ValueError(f"Bad anchor_time '{anchor_time}', expected 'HH:MM'") from e

    first_day = idx_local[0].normalize()
    anchor0 = first_day + pd.Timedelta(hours=hh, minutes=mm)

    # If our first timestamp is already after today's anchor, keep today's anchor;
    # else jump to the next day's anchor.
    target = anchor0 if idx_local[0] <= anchor0 else (anchor0 + pd.Timedelta(days=1))

    # position of first index >= target
    pos = idx_local.get_indexer([target], method="bfill")[0]
    if pos == -1:
        return df, None

    df2 = df.iloc[pos:].copy()
    return df2, df.index[pos]


def time_split(df: pd.DataFrame,
               train_end: pd.Timestamp,
               val_end: pd.Timestamp | None,
               time_col: str | None = None):
    df = ensure_datetime_index(df, time_col=time_col, utc=True)

    # harmonise les timezones des bornes
    train_end = pd.to_datetime(train_end, utc=True)
    val_end = pd.to_datetime(val_end, utc=True) if val_end is not None else None

    train = df.loc[:train_end]
    if val_end is None:
        val = df.iloc[0:0]
        test = df.loc[train_end:]
    else:
        val = df.loc[train_end:val_end]
        test = df.loc[val_end:]
    return train, val, test

def window_indices(n: int, win: int, step: int) -> List[Tuple[int, int]]:
    idx = []
    start = 0
    while start + win <= n:
        idx.append((start, start + win))
        start += step
    return idx

def _freq_info(idx):
    diffs = idx.to_series().diff().dropna()
    approx = diffs.median() if not diffs.empty else pd.NaT
    try:
        inf = pd.infer_freq(idx)
    except Exception:
        inf = None
    print(f"[FREQ] after resample: infer_freq={inf} | median_step={approx} | n={len(idx)} | first={idx.min()} | last={idx.max()}")


# ------------------------------------------------------
# 5) Image generation
# ------------------------------------------------------

# Moved to another file

# -----------------------------------------------------
# 6) Assembly: labels generation + images & CSV
# -----------------------------------------------------

def build_dataset(csv_path: Path,
                  out_dir: Path,
                  symbol: str,
                  horizon_list: List[int] = [1, 2],
                  median_window: int = 100,
                  rs_rule: str | None = None, 
                  rs_label: str = "left",
                  rs_closed: str = "left", 
                  rs_dropna: bool = True,
                  deseason_mode: str = "none",
                  deseason_bucket: str = "hour",
                  image_windows: Optional[List[int]] = None,  # ex: [64, 96, 128]
                  image_step: int = 1,
                  splits: Optional[Tuple[str, Optional[str]]] = None,  # ("2024-12-31","2025-03-31")
                  image_encoder: str = "heatmap",
                  ts_ma_window: int = 60,
                  img_w: int = 256, img_h: int = 256, img_dpi: int = 100,
                  align_enabled: bool = False,
                  align_time_str: str = "14:00",
                  align_tz: str = "UTC",
                  heatmap_cmap: str = "viridis",
                  heatmap_vmin: float | None = None,
                  heatmap_vmax: float | None = None,
                  day_sep_enabled: bool = False,
                  day_sep_label: bool = True,
                  day_sep_color: str = "#888888",
                  day_sep_alpha: float = 0.35,
                  day_sep_lw: float = 0.8,
                  day_label_kind: str = "number",   # number | abbr | name
                  show_ma_top: bool = True,
                  bb_enabled: bool = False,
                  bb_window: int = 20,
                  bb_nstd: float = 2.0,
                  bottom_panel: str = "volume",   # volume|rsi|none
                  rsi_window: int = 14,
                  rsi_source: int = 'vol',
                  fg: str = "white",
                  bg: str = "black",
                  rp_series: str = "vol",          # "vol" | "close" | <any column name in part_df/ohlcv>
                  rp_normalize: str = "zscore",    # "zscore" | "minmax" | "none"
                  rp_metric: str = "euclidean",    # reserved for future
                  rp_epsilon_mode: str = "quantile",  # "quantile" | "fixed" | "none"
                  rp_epsilon_q: float = 0.10,
                  rp_epsilon_value: float | None = None,
                  rp_binarize: bool = True,
                  rp_cmap: str = "gray",
                  rp_invert: bool = True,
                  gaf_mode: str = "gasf",            # 'gasf' | 'gadf'
                  gaf_normalize: str = "minmax",     # 'minmax' | 'zscore' | 'none'
                  gaf_cmap: str = "viridis",
                  gaf_invert: bool = False,
                  out_path_batch: str = None,
                  label_mode: str = "standard"       # "standard" | "overnight_gap"
                  ) -> None:

    df = load_ohlcv(
        csv_path,
        tz_utc=m.tz_utc if "m" in globals() else True,   # si tu appelles avec m; sinon passe les args reçus
        time_col=m.time_col if "m" in globals() else None,
        rs_rule=rs_rule,
        rs_label=rs_label,
        rs_closed=rs_closed,
        rs_dropna=rs_dropna,
    )
    _freq_info(df.index)
    ohlcv = df[["open","high","low","close"] + ([ "volume"] if "volume" in df.columns else [])].copy()
    vol_raw = garman_klass_sigma(df).sort_index()

    # Decide on de-seasonalization
    if deseason_mode == "none":
        vol_src = vol_raw
        
    elif deseason_mode == "robust_train":
        if splits is None or splits[0] is None:
            raise ValueError("robust_train de-seasonalization requires --train_end to fit stats on train only.")
        train_end = pd.to_datetime(splits[0], utc=True)
        stats = vol_smile.fit_robust_tod_stats(vol_raw.loc[:train_end], bucket=deseason_bucket)
        vol_src = vol_smile.apply_robust_tod_zscore(vol_raw, stats, bucket=deseason_bucket)

    elif deseason_mode == "robust_expanding":
        vol_src = vol_smile.deseason_robust_expanding(vol_raw, bucket=deseason_bucket)

    else:
        raise ValueError("Unknown deseason_mode")
    
    ind = compute_global_indicators(
        vol=vol_src,
        ohlcv=ohlcv,
        ma_window=ts_ma_window,
        bb_window=bb_window,     # ← depuis YAML (images.ts_overlays.top.bb_window)
        bb_nstd=bb_nstd,         # ← depuis YAML (images.ts_overlays.top.bb_nstd)
        rsi_window=rsi_window,   # ← depuis YAML (images.ts_overlays.bottom.rsi_window)
        rsi_source=rsi_source,   # ← depuis YAML (vol|close)
    )
   
    meta_all = []
    
    # Handle different labeling modes
    if label_mode == "overnight_gap":
        # For overnight mode, we only use horizon=1 (overnight gap)
        if 1 not in horizon_list:
            print(f"[WARNING] overnight_gap mode requires horizon=1, but got {horizon_list}. Using horizon=1.")
            horizon_list = [1]
        
        # Generate overnight labels
        lab = make_labels_overnight(ohlcv, median_window=median_window, exclude_weekends=True)
        lab["symbol"] = symbol
        lab["horizon"] = 1  # Always 1 for overnight gap
        
        # For overnight mode, we need to align images to session close times
        # The labels are indexed by close timestamps, so we need to create images
        # that end at those close times and cover the full session (09:30-16:00)
        
    else:
        # Standard mode: use the original make_labels function
        for h in horizon_list:
            lab = make_labels(vol_src, horizon_days=h, median_window=median_window, drop_na=True)
            lab["symbol"] = symbol
            lab["horizon"] = h

    # Temporal split (applies to both modes)
    if splits is not None:
        train_end = pd.to_datetime(splits[0], utc=True)
        val_end = pd.to_datetime(splits[1], utc=True) if splits[1] else None
        train, val, test = time_split(lab, train_end, val_end)
        parts = [("train", train), ("val", val), ("test", test)]
    else:
        parts = [("full", lab)]
            
    if align_enabled:
        aligned_parts = []
        for split_name, part_df in parts:
            if part_df.empty:
                aligned_parts.append((split_name, part_df))
                continue
            part_aligned, anchor_ts = realign_to_day_anchor(
                part_df, anchor_time=align_time_str, anchor_tz=align_tz
            )
            kept = len(part_aligned); drop = len(part_df) - kept
            print(f"[ALIGN] split={split_name} anchor={align_time_str} {align_tz} "
                f"dropped={drop} kept={kept} first={anchor_ts}")
            aligned_parts.append((split_name, part_aligned))
        parts = aligned_parts

    # Images
    if image_windows:
        for split_name, part_df in parts:
            if part_df.empty:
                continue
                
            if label_mode == "overnight_gap":
                # For overnight mode, create one image per close timestamp
                # Each image covers the full session ending at that close time
                for close_ts in tqdm(
                    part_df.index,
                    desc=f"Generating {image_encoder} images | {symbol} | {split_name} | overnight",
                    dynamic_ncols=True,
                    leave=False
                ):
                    # Find the session window ending at this close time
                    # Convert to market timezone to find session boundaries
                    if ohlcv.index.tz is None:
                        ohlcv_tz = ohlcv.index.tz_localize("UTC")
                    else:
                        ohlcv_tz = ohlcv.index.tz_convert("America/New_York")
                    
                    close_ts_market = close_ts.tz_convert("America/New_York")
                    session_date = close_ts_market.date()
                    
                    # Find session start (09:30) and end (16:00) for this date
                    session_start = pd.Timestamp(f"{session_date} 09:30", tz="America/New_York")
                    session_end = pd.Timestamp(f"{session_date} 16:00", tz="America/New_York")
                    
                    # Convert back to original timezone for indexing
                    if ohlcv.index.tz is None:
                        session_start = session_start.tz_convert("UTC").tz_localize(None)
                        session_end = session_end.tz_convert("UTC").tz_localize(None)
                    else:
                        session_start = session_start.tz_convert(ohlcv.index.tz)
                        session_end = session_end.tz_convert(ohlcv.index.tz)
                    
                    # Get session data
                    session_data = ohlcv.loc[session_start:session_end]
                    if len(session_data) < 10:  # Need minimum bars for a meaningful image
                        continue
                    
                    y = int(part_df.loc[close_ts, "y_h1"])
                    fname = ts_to_filename(close_ts)
                    out_path = Path(out_path_batch) / f"y{y}" / f"{fname}.png" if out_path_batch else out_dir / symbol / split_name / f"h1" / f"y{y}" / f"{fname}.png"
                    
                    # Generate image for this session
                    if image_encoder == "ts_ohlc":
                        to_timeseries_image(session_data, out_path, kind="ohlc",
                                          ma_window=ts_ma_window,
                                          width_px=img_w, height_px=img_h, dpi=img_dpi)
                    elif image_encoder == "ts_vol":
                        # For ts_vol, we need to compute volatility for the session
                        session_vol = garman_klass_sigma(session_data)
                        session_vol_df = pd.DataFrame(index=session_data.index)
                        session_vol_df["vol"] = session_vol
                        if "volume" in session_data.columns:
                            session_vol_df["volume"] = session_data["volume"]
                        
                        # Get precomputed indicators for this session
                        session_ind = ind.loc[session_data.index]
                        
                        to_timeseries_image(
                            session_vol_df, out_path, kind="vol", vol_col="vol",
                            ma_window=ts_ma_window,
                            # overlays
                            show_ma_top=show_ma_top,
                            show_bbands=bb_enabled,
                            bb_window=bb_window, bb_nstd=bb_nstd,
                            bottom_panel=bottom_panel, rsi_window=rsi_window,
                            # day separators
                            show_day_separators=day_sep_enabled,
                            day_sep_label=day_sep_label,
                            day_sep_color=day_sep_color,
                            day_sep_alpha=day_sep_alpha,
                            day_sep_lw=day_sep_lw,
                            day_label_kind=day_label_kind,
                            # provide precomputed series
                            ma_series=session_ind["vol_ma"],
                            bb_up_series=session_ind["bb_up"],
                            bb_lo_series=session_ind["bb_lo"],
                            rsi_series=session_ind["rsi"] if bottom_panel == "rsi" else None,
                            # style/size
                            fg=fg, bg=bg, width_px=img_w, height_px=img_h, dpi=img_dpi
                        )
                    else:
                        print(f"[WARNING] overnight_gap mode supports ts_ohlc and ts_vol encoders, got {image_encoder}")
                        continue
                        
            else:
                # Standard mode: use window indices
                for h in horizon_list:
                    idx_pairs = window_indices(len(part_df), win=max(image_windows), step=image_step)
                    for (a, b) in tqdm(
                        idx_pairs,
                        desc=f"Generating {image_encoder} images | {symbol} | {split_name} | horizon={h}d",
                        total=len(idx_pairs),
                        dynamic_ncols=True,
                        leave=False
                    ):
                        idx = part_df.index[a:b]
                        end_ts = idx[-1]
                        y = int(part_df.loc[end_ts, f"y_h{h}"])
                        fname = ts_to_filename(end_ts)
                        out_path = Path(out_path_batch) / f"y{y}" / f"{fname}.png" if out_path_batch else out_dir / symbol / split_name / f"h{h}" / f"y{y}" / f"{fname}.png"

                    if image_encoder == "heatmap":
                        # features simples pour heatmap
                        win_df = part_df.loc[idx, ["vol","median_hist"]]
                        to_heatmap_image(win_df.T, out_path,
                                        cmap=heatmap_cmap,
                                        vmin=heatmap_vmin, vmax=heatmap_vmax)

                    elif image_encoder == "ts_vol":
                        win_df = pd.DataFrame(index=idx)
                        win_df["vol"] = vol_src.reindex(idx)
                        if "volume" in ohlcv.columns:
                            win_df["volume"] = ohlcv["volume"].reindex(idx)
                        if win_df["vol"].isna().any():
                            continue

                        # slice precomputed indicators for this exact window
                        win_ind = ind.loc[idx]
                        to_timeseries_image(
                            win_df, out_path, kind="vol", vol_col="vol",
                            ma_window=ts_ma_window,
                            # overlays (toggles de ta YAML)
                            show_ma_top=show_ma_top,
                            show_bbands=bb_enabled,
                            bb_window=bb_window, bb_nstd=bb_nstd,
                            bottom_panel=bottom_panel, rsi_window=rsi_window,
                            #day sep
                            show_day_separators=day_sep_enabled,
                            day_sep_label=day_sep_label,
                            day_sep_color=day_sep_color,
                            day_sep_alpha=day_sep_alpha,
                            day_sep_lw=day_sep_lw,
                            day_label_kind=day_label_kind,
                            # --- provide precomputed series ---
                            ma_series=win_ind["vol_ma"],
                            bb_up_series=win_ind["bb_up"],
                            bb_lo_series=win_ind["bb_lo"],
                            rsi_series=win_ind["rsi"] if bottom_panel == "rsi" else None,
                            # style/size
                            fg=fg, bg=bg, width_px=img_w, height_px=img_h, dpi=img_dpi
                        )

                    elif image_encoder == "ts_ohlc":
                        needed = ["open","high","low","close"]
                        if not set(needed).issubset(ohlcv.columns):
                            continue  # pas d'OHLC -> skip
                        win_df = ohlcv.reindex(idx)
                        if win_df[needed].isna().any().any():
                            continue
                        to_timeseries_image(win_df, out_path, kind="ohlc",
                                            ma_window=ts_ma_window,
                            width_px=img_w, height_px=img_h, dpi=img_dpi)
                    
                    elif image_encoder == "recurrence":
                        # choose the series to plot
                        if rp_series == "vol":
                            s = vol_src.reindex(idx)
                        elif rp_series == "close":
                            s = ohlcv["close"].reindex(idx) if "close" in ohlcv.columns else None
                        else:
                            # try from part_df first (e.g., "median_hist"), then from ohlcv
                            s = part_df[rp_series].reindex(idx) if rp_series in part_df.columns else (
                                ohlcv[rp_series].reindex(idx) if rp_series in ohlcv.columns else None
                            )
                        if s is None or s.isna().any():
                            continue

                        fname = ts_to_filename(end_ts)
                        out_path = Path(out_path_batch) / split_name / f"h{h}" / f"y{y}" / f"{fname}.png" if out_path_batch else out_dir / symbol / split_name / f"h{h}" / f"y{y}" / f"{fname}.png"

                        to_recurrence_image(
                            s, out_path,
                            normalize=rp_normalize,
                            metric=rp_metric,
                            epsilon_mode=rp_epsilon_mode,
                            epsilon_q=rp_epsilon_q,
                            epsilon_value=rp_epsilon_value,
                            binarize=rp_binarize,
                            cmap=rp_cmap,
                            invert=rp_invert
                        )
                    
                    elif image_encoder == "gaf":
                        # choose the 1D series to encode (default: volatility)
                        s = vol_src.reindex(idx)
                        if s.isna().any():
                            continue
                        fname = ts_to_filename(end_ts)
                        out_path = Path(out_path_batch) / split_name / f"h{h}" / f"y{y}" / f"{fname}.png" if out_path_batch else out_dir / symbol / split_name / f"h{h}" / f"y{y}" / f"{fname}.png"
                        to_gaf_image(
                            s, out_path,
                            mode=gaf_mode,
                            normalize=gaf_normalize,
                            cmap=gaf_cmap,
                            invert=gaf_invert
                        )

        else:
            # If no images, we save a CSV of labels per split
            for split_name, part_df in parts:
                if part_df.empty:
                    continue
                out_csv = out_dir / symbol / split_name / f"labels_h{part_df['horizon'].iloc[0]}.csv"
                print(out_csv, "out csv")
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                part_df.to_csv(out_csv)

    # Save
    if meta_all:
        meta_df = pd.DataFrame(meta_all)
        meta_csv = out_dir / symbol / "metadata_images.csv"
        meta_csv.parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(meta_csv, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Build volatility image dataset with GK labels.")
    p.add_argument("--config", type=str, default="./config/dataset.yaml", help="Path to a YAML config file.")
    p.add_argument("--csv", type=Path, default=None, help="Chemin vers un CSV OHLCV")
    p.add_argument("--out", type=Path, default=None, help="Dossier de sortie")
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--horizons", type=int, nargs="+", default=None, help="Ex: 1 2")
    p.add_argument("--median_window", type=int, default=None)
    p.add_argument("--image_windows", type=int, nargs="*", default=None, help="Taille(s) de fenêtre pour images")
    p.add_argument("--image_step", type=int, default=None)
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--val_end", type=str, default=None)
    p.add_argument("--image_encoder", choices=["heatmap","ts_vol","ts_ohlc"], default=None,help="Type d’images à générer.")
    p.add_argument("--ts_ma_window", type=int, default=None, help="MA pour les images time-series.")
    p.add_argument("--img_w", type=int, default=None)
    p.add_argument("--img_h", type=int, default=None)
    p.add_argument("--img_dpi", type=int, default=None)
    p.add_argument("--deseason_mode", choices=["none","robust_train","robust_expanding"], default=None,
               help="De-seasonalization mode for volatility: none, robust fit-on-train, or strict expanding ex-ante.")
    p.add_argument("--deseason_bucket", choices=["minute","hour"], default=None,
               help="Time-of-day bucket granularity for de-seasonalization.")
    p.add_argument("--time_col", type=str, default=None,
               help="Name of the timestamp column in the CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    m = merge_args_with_config(args)
    clean_scope = getattr(m, "clean_scope", "symbol")   # par défaut on nettoie le symbole courant
    do_clean    = getattr(m, "clean", True)             # active/désactive via YAML/CLI

    if do_clean and clean_scope:
        n = indicators.clean_dataset_dir(Path(m.out), symbol=m.symbol, what=clean_scope)
        print(f"[CLEAN] Removed {n} item(s) under '{clean_scope}' before generation.")
    splits = None
    if m.train_end:
        splits = (m.train_end, m.val_end)
    build_dataset(csv_path=Path(m.csv),
                out_dir=Path(m.out),
                symbol=m.symbol,
                horizon_list=m.horizons,
                median_window=m.median_window,
                image_windows=m.image_windows,
                image_step=m.image_step,
                splits=(m.train_end, m.val_end) if m.train_end else None,
                deseason_mode=m.deseason_mode,
                deseason_bucket=m.deseason_bucket,
                image_encoder=m.image_encoder,
                ts_ma_window=m.ts_ma_window,
                img_w=m.img_w, img_h=m.img_h, img_dpi=m.img_dpi,
                rs_rule=m.rs_rule, rs_label=m.rs_label, rs_closed=m.rs_closed, rs_dropna=m.rs_dropna,
                align_enabled=m.align_enabled,
                align_time_str=m.align_time_str,
                align_tz=m.align_tz,
                heatmap_cmap=m.heatmap_cmap,
                heatmap_vmin=m.heatmap_vmin,
                heatmap_vmax=m.heatmap_vmax,
                day_sep_enabled=m.day_sep_enabled,
                day_sep_label=m.day_sep_label,
                day_sep_color=m.day_sep_color,
                day_sep_alpha=m.day_sep_alpha,
                day_sep_lw=m.day_sep_lw,
                day_label_kind=m.day_label_kind,
                show_ma_top=m.show_ma_top, bb_enabled=m.bb_enabled, bb_window=m.bb_window, bb_nstd=m.bb_nstd,
                bottom_panel=m.bottom_panel, rsi_window=m.rsi_window, rsi_source=m.rsi_source,
                fg=m.fg, bg=m.bg,     
                rp_series=m.rp_series,
                rp_normalize=m.rp_normalize,
                rp_metric=m.rp_metric,
                rp_epsilon_mode=m.rp_epsilon_mode,
                rp_epsilon_q=m.rp_epsilon_q,
                rp_epsilon_value=m.rp_epsilon_value,
                rp_binarize=m.rp_binarize,
                rp_cmap=m.rp_cmap,
                rp_invert=m.rp_invert,
                gaf_mode=m.gaf_mode,
                gaf_normalize=m.gaf_normalize,
                gaf_cmap=m.gaf_cmap,
                gaf_invert=m.gaf_invert,
                label_mode=m.label_mode,
            )
