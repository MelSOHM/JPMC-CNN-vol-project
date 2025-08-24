#!/usr/bin/env python3
# build_dataset.py
# Dataset builder: OHLCV -> GK vol -> labels (t+1/t+2) -> (optionnel) images
# Author: Mel & Daisy (JPMC project)

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt


# ---------------------------
# 1) Loading & Prep
# ---------------------------

def load_ohlcv(csv_path: Path,
               tz_utc: bool = True,
               parse_dates_col: str = "timestamp") -> pd.DataFrame:
    """
    Expect a CSV with columns: timestamp, open, high, low, close, volume.
    The timstamp must be sorted in increasing order
    """
    df = pd.read_csv(csv_path)
    if parse_dates_col in df.columns:
        df[parse_dates_col] = pd.to_datetime(df[parse_dates_col], utc=tz_utc)
        df = df.sort_values(parse_dates_col)
        df = df.set_index(parse_dates_col)
    req = {"open", "high", "low", "close"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Colonnes manquantes: {req - set(df.columns)}")
    return df


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

# ---------------------------------------------------------
# 3) Get Rid of intraday Volatility Smile
# ---------------------------------------------------------

def intraday_deseasonalize(vol: pd.Series,
                           method: str = "median",
                           index_is_datetime: bool = True) -> pd.Series:
    """
    Decrease intraday seasonal effect
    Compute a minute/time of day factor via median and mean, then divide
    """
    if not index_is_datetime:
        return vol

    # Exemple pour séries horaires: groupby par (heure, minute)
    tod = list(zip(vol.index.hour, vol.index.minute))
    gb = pd.Series(tod, index=vol.index)
    df = pd.DataFrame({"vol": vol, "tod": gb})
    if method == "median":
        f = df.groupby("tod")["vol"].median()
    else:
        f = df.groupby("tod")["vol"].mean()

    def norm_one(ts):
        key = (ts.hour, ts.minute)
        denom = f.get(key, np.nan)
        if denom and denom > 0:
            return ts
        return ts

    norm = vol.copy()
    for idx, val in vol.items():
        denom = f.get((idx.hour, idx.minute), np.nan)
        if pd.notna(denom) and denom > 0:
            norm.loc[idx] = val / denom
        else:
            norm.loc[idx] = np.nan
    return norm.rename(vol.name + "_deseason")


# -----------------------------------------------
# 4) Rolling Median and ex-ante t+h labels
# -----------------------------------------------

def rolling_median_ex_ante(x: pd.Series, window: int) -> pd.Series:
    """
    Rolling median only based on historical data (excluding today).
    """
    return x.shift(1).rolling(window=window, min_periods=window).median()


def make_labels(vol: pd.Series,
                horizon: int = 1,
                median_window: int = 100,
                drop_na: bool = True) -> pd.DataFrame:
    """
    Label y_t = 1{ vol_{t+h} > median_{t} } ; median_t = Rolling ex-ante median.
    - horizon: t+1 or t+2 (etc.)
    - median_window: Window lenght for historical median.
    """
    med = rolling_median_ex_ante(vol, window=median_window).rename("median_hist")
    y = (vol.shift(-horizon) > med).astype("float").rename(f"y_h{horizon}")
    out = pd.concat({"vol": vol, "median_hist": med, f"y_h{horizon}": y}, axis=1)
    if drop_na:
        out = out.dropna()
    return out

# -------------------------------------------------
# 5) (train/val/test) temporal split 
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


# ------------------------------------------------------
# 6) Image generation - Heatmap
# ------------------------------------------------------

def window_indices(n: int, win: int, step: int) -> List[Tuple[int, int]]:
    idx = []
    start = 0
    while start + win <= n:
        idx.append((start, start + win))
        start += step
    return idx

def to_heatmap_image(window_df: pd.DataFrame,
                     out_path: Path,
                     cmap: str = "gray",
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None):
    """
    Encode a (features x temps) bloc into a heatmap (grayscale by default)
    Each feature is scaled into [0,1] (min-max) before stacking.
    """
    # Normalize each feature in the window
    X = []
    for col in window_df.columns:
        v = window_df[col].values.astype(float)
        vmin_c = np.nanmin(v) if vmin is None else vmin
        vmax_c = np.nanmax(v) if vmax is None else vmax
        if np.isfinite(vmin_c) and np.isfinite(vmax_c) and vmax_c > vmin_c:
            v_norm = (v - vmin_c) / (vmax_c - vmin_c)
        else:
            v_norm = np.zeros_like(v)
        X.append(v_norm)
    # Matrix (features x time)
    M = np.vstack(X)
    plt.figure()
    plt.imshow(M, aspect="auto", cmap=cmap, origin="lower")  # 2D image
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# -----------------------------------------------------
# 7) Assembly: labels generation + images & CSV
# -----------------------------------------------------

def build_dataset(csv_path: Path,
                  out_dir: Path,
                  symbol: str,
                  horizon_list: List[int] = [1, 2],
                  median_window: int = 100,
                  deseason: bool = False,
                  image_windows: Optional[List[int]] = None,  # ex: [64, 96, 128]
                  image_step: int = 1,
                  splits: Optional[Tuple[str, Optional[str]]] = None  # ("2024-12-31","2025-03-31")
                  ) -> None:

    df = load_ohlcv(csv_path)
    vol = garman_klass_sigma(df) # vol by interval (GK)
    if deseason:
        vol = intraday_deseasonalize(vol).dropna()

    meta_all = []

    for h in horizon_list:
        lab = make_labels(vol, horizon=h, median_window=median_window, drop_na=True)
        lab["symbol"] = symbol
        lab["horizon"] = h

        # Temporal split
        if splits is not None:
            train_end = pd.to_datetime(splits[0], utc=True)
            val_end = pd.to_datetime(splits[1], utc=True) if splits[1] else None
            train, val, test = time_split(lab, train_end, val_end, time_col="ts_event")
            parts = [("train", train), ("val", val), ("test", test)]
        else:
            parts = [("full", lab)]

        # Images
        if image_windows:
            for split_name, part_df in parts:
                if part_df.empty:
                    continue
                # base features for the image: [vol, median_hist]
                F = part_df[["vol", "median_hist"]].copy()
                idx_pairs = window_indices(len(F), win=max(image_windows), step=image_step)

                for (a, b) in idx_pairs:
                    win_df = F.iloc[a:b]
                    # Associated label to right edge of the window (b-1)
                    end_ts = F.index[b - 1]
                    row = part_df.loc[end_ts]
                    y = int(row[f"y_h{h}"])
                    # File: out_dir/symbol/split/h{h}/label/
                    out_path = out_dir / symbol / split_name / f"h{h}" / f"y{y}" / f"{end_ts.value}.png"
                    to_heatmap_image(win_df.T, out_path)  # (features x time)
                    meta_all.append({
                        "symbol": symbol,
                        "split": split_name,
                        "horizon": h,
                        "end_ts": end_ts,
                        "img_path": str(out_path),
                        "label": y,
                        "vol_tplus_h": part_df.loc[end_ts, "vol"],
                        "median_hist_at_t": part_df.loc[end_ts, "median_hist"]
                    })
        else:
            # If no images, we save a CSV of labels per split
            for split_name, part_df in parts:
                if part_df.empty:
                    continue
                out_csv = out_dir / symbol / split_name / f"labels_h{h}.csv"
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
    p.add_argument("--csv", type=Path, required=True, help="Chemin vers un CSV OHLCV")
    p.add_argument("--out", type=Path, required=True, help="Dossier de sortie")
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2], help="Ex: 1 2")
    p.add_argument("--median_window", type=int, default=100)
    p.add_argument("--deseason", action="store_true", help="Normalisation intra-day")
    p.add_argument("--image_windows", type=int, nargs="*", default=None, help="Taille(s) de fenêtre pour images")
    p.add_argument("--image_step", type=int, default=1)
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--val_end", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    splits = None
    if args.train_end:
        splits = (args.train_end, args.val_end)
    build_dataset(csv_path=args.csv,
                  out_dir=args.out,
                  symbol=args.symbol,
                  horizon_list=args.horizons,
                  median_window=args.median_window,
                  deseason=args.deseason,
                  image_windows=args.image_windows,
                  image_step=args.image_step,
                  splits=splits)
