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
# 6) Image generation
# ------------------------------------------------------

# Kind : Heatmap 

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

# Kind: Time series

def to_timeseries_image(
    window_df: pd.DataFrame,
    out_path: Path,
    *,
    kind: str = "vol",              # "vol" | "ohlc"
    ma_window: int = 60,            # Moving-average window size
    price_cols=("open","high","low","close"),
    volume_col: str = "volume",
    vol_col: str = "vol",           # Volatility column
    fg: str = "white",
    bg: str = "black",
    width_px: int = 256,
    height_px: int = 256,
    dpi: int = 100,
):
    """
    Generate a panel image (top: main series + moving average; bottom: bars).
    - kind="ohlc": line-style candlesticks + MA(close); bars = volume if available.
    - kind="vol" : volatility line + MA(vol); bars = volume if available, otherwise volatility bars.

    Requirements:
        * kind="ohlc": window_df must contain open, high, low, close (ideally also volume).
        * kind="vol" : window_df must contain the `vol_col` column (and volume if desired).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Figure and axes (80% top / 20% bottom) ----
    fig_h = height_px / dpi
    fig_w = width_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(nrows=5, ncols=1, hspace=0.0)
    ax_top = fig.add_subplot(gs[:-1, 0])                 # 4/5 height
    ax_bot = fig.add_subplot(gs[-1, 0], sharex=ax_top)  # 1/5 height

    fig.patch.set_facecolor(bg)
    ax_top.set_facecolor(bg)
    ax_bot.set_facecolor(bg)
    for ax in (ax_top, ax_bot):
        ax.axis("off")
        ax.margins(x=0)

    x = np.arange(len(window_df))

    if kind == "ohlc":
        o, h, l, c = [window_df[col].values.astype(float) for col in price_cols]
        # y-limits with a small padding
        ymin = np.nanmin(l); ymax = np.nanmax(h)
        pad = 0.03 * (ymax - ymin if ymax > ymin else 1.0)
        ax_top.set_ylim(ymin - pad, ymax + pad)

        # --- line-style candlesticks (as in the sample paper) ---
        ax_top.vlines(x, l, h, colors=fg, linewidth=1.0)
        ax_top.hlines(o, x - 0.25, x,        colors=fg, linewidth=1.0)  # open tick on the left
        ax_top.hlines(c, x,        x + 0.25, colors=fg, linewidth=1.0)  # close tick on the right

        # Moving average on close
        c_ma = pd.Series(c).rolling(ma_window, min_periods=1).mean().values
        ax_top.plot(x, c_ma, color=fg, linewidth=1.4, alpha=0.85)

        # Bars (volume if available)
        bars = window_df[volume_col].values.astype(float) if volume_col in window_df else np.zeros_like(x)
        ax_bot.bar(x, bars, color=fg, width=0.8)

    else:  # kind == "vol"
        v = window_df[vol_col].values.astype(float)
        v_ma = pd.Series(v).rolling(ma_window, min_periods=1).mean().values

        # y-limits with padding
        ymin = np.nanmin(np.minimum(v, v_ma))
        ymax = np.nanmax(np.maximum(v, v_ma))
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        ax_top.set_ylim(ymin - pad, ymax + pad)

        # Vol curve + moving average
        ax_top.plot(x, v,    color=fg, linewidth=1.0)
        ax_top.plot(x, v_ma, color=fg, linewidth=1.6, alpha=0.8)

        # Bars: volume if present, otherwise "volatility bars"
        if volume_col in window_df:
            bars = window_df[volume_col].values.astype(float)
        else:
            bars = v
        ax_bot.bar(x, bars, color=fg, width=0.8)

    plt.savefig(out_path, facecolor=bg, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

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
                  splits: Optional[Tuple[str, Optional[str]]] = None,  # ("2024-12-31","2025-03-31")
                  image_encoder: str = "heatmap",
                  ts_ma_window: int = 60,
                  img_w: int = 256, img_h: int = 256, img_dpi: int = 100) -> None:

    df = load_ohlcv(csv_path)
    ohlcv = df[["open","high","low","close"] + ([ "volume"] if "volume" in df.columns else [])].copy()
    vol = garman_klass_sigma(df) # vol by interval (GK)
    if not isinstance(vol.index, pd.DatetimeIndex):
        vol.index = pd.to_datetime(vol.index, utc=True, errors="coerce")
    vol = vol.sort_index()
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
                idx_pairs = window_indices(len(part_df), win=max(image_windows), step=image_step)

                for (a, b) in idx_pairs:
                    # fenêtre d’index temporels alignée
                    idx = part_df.index[a:b]
                    end_ts = idx[-1]
                    y = int(part_df.loc[end_ts, f"y_h{h}"])
                    out_path = out_dir / symbol / split_name / f"h{h}" / f"y{y}" / f"{end_ts.value}.png"

                    if image_encoder == "heatmap":
                        # features simples pour heatmap
                        win_df = part_df.loc[idx, ["vol","median_hist"]]
                        to_heatmap_image(win_df.T, out_path, mode="colormap", cmap="viridis")  # ta fonction heatmap existante

                    elif image_encoder == "ts_vol":
                        # panel volatilité (vol + MA + barres)
                        win_df = pd.DataFrame(index=idx)
                        win_df["vol"] = vol.reindex(idx)  # vol est le Series retourné par garman_klass_sigma
                        if "volume" in ohlcv.columns:
                            win_df["volume"] = ohlcv["volume"].reindex(idx)
                        if win_df["vol"].isna().any():
                            continue  # skip fenêtres incomplètes
                        to_timeseries_image(win_df, out_path, kind="vol",
                                            vol_col="vol", ma_window=ts_ma_window,
                                            width_px=img_w, height_px=img_h, dpi=img_dpi)

                    elif image_encoder == "ts_ohlc":
                        # panel chandeliers (OHLC + MA(close) + volume)
                        needed = ["open","high","low","close"]
                        if not set(needed).issubset(ohlcv.columns):
                            continue  # pas d'OHLC -> skip
                        win_df = ohlcv.reindex(idx)
                        if win_df[needed].isna().any().any():
                            continue
                        to_timeseries_image(win_df, out_path, kind="ohlc",
                                            ma_window=ts_ma_window,
                            width_px=img_w, height_px=img_h, dpi=img_dpi)

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
    p.add_argument("--image_encoder", choices=["heatmap","ts_vol","ts_ohlc"], default="heatmap",help="Type d’images à générer.")
    p.add_argument("--ts_ma_window", type=int, default=60, help="MA pour les images time-series.")
    p.add_argument("--img_w", type=int, default=256)
    p.add_argument("--img_h", type=int, default=256)
    p.add_argument("--img_dpi", type=int, default=100)
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
                  splits=splits,
                  image_encoder=args.image_encoder,
                  ts_ma_window=args.ts_ma_window,
                  img_w=args.img_w, img_h=args.img_h, img_dpi=args.img_dpi)
