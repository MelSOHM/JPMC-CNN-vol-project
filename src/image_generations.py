from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import indicators
from tqdm.auto import tqdm

# Kind : Heatmap 

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
    kind: str = "vol",
    ma_window: int = 60,
    price_cols=("open","high","low","close"),
    volume_col: str = "volume",
    vol_col: str = "vol",
    # overlays
    show_ma_top: bool = True,
    show_bbands: bool = False,
    bb_window: int = 20,
    bb_nstd: float = 2.0,
    bottom_panel: str = "volume",   # volume | rsi | none
    rsi_window: int = 14,
    # style
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
    fig_h = height_px / dpi; fig_w = width_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(nrows=5, ncols=1, hspace=0.0)
    ax_top = fig.add_subplot(gs[:-1, 0])                 # 4/5 height
    ax_bot = fig.add_subplot(gs[-1, 0], sharex=ax_top)  # 1/5 height

    fig.patch.set_facecolor(bg); ax_top.set_facecolor(bg); ax_bot.set_facecolor(bg)
    for ax in (ax_top, ax_bot): ax.axis("off"); ax.margins(x=0)

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
        v = pd.Series(window_df[vol_col].values.astype(float), index=window_df.index)
        v_ma = v.rolling(ma_window, min_periods=1).mean().values if show_ma_top else None

        # y-limits include overlays if any
        ymin, ymax = v.min(), v.max()
        if show_ma_top and np.isfinite(np.nanmax(v_ma)):
            ymin = min(ymin, np.nanmin(v_ma)); ymax = max(ymax, np.nanmax(v_ma))
        if show_bbands:
            bb_mid, bb_up, bb_lo = indicators.bollinger_bands(v, window=bb_window, nstd=bb_nstd, min_periods=1)
            ymin = min(ymin, float(bb_lo.min())); ymax = max(ymax, float(bb_up.max()))
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        ax_top.set_ylim(ymin - pad, ymax + pad)

        # main line
        ax_top.plot(x, v.values, color=fg, linewidth=1.0)
        if show_ma_top:
            ax_top.plot(x, v_ma, color=fg, linewidth=1.6, alpha=0.85)
        if show_bbands:
            ax_top.plot(x, bb_up.values, color=fg, linewidth=0.9, alpha=0.6)
            ax_top.plot(x, bb_lo.values, color=fg, linewidth=0.9, alpha=0.6)
            # optional mid:
            # ax_top.plot(x, bb_mid.values, color=fg, linewidth=0.8, alpha=0.4, linestyle="--")

        # bottom panel selection
        bp = (bottom_panel or "volume").lower()
        if bp == "rsi":
            rsi = indicators.rsi_wilder(v, window=rsi_window)
            ax_bot.set_ylim(0, 100)
            ax_bot.plot(x, rsi.values, color=fg, linewidth=1.2)
            ax_bot.hlines([30,70], 0, len(x)-1, colors=fg, linewidth=0.8, alpha=0.4, linestyles="dashed")
        elif bp == "volume" and "volume" in window_df:
            ax_bot.bar(x, window_df[volume_col].values.astype(float), color=fg, width=0.8)
        # elif bp == "none": do nothing

    plt.savefig(out_path, facecolor=bg, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    

# Kind : Reccurence Plot

def to_recurrence_image(
    series: pd.Series,
    out_path: Path,
    *,
    normalize: str = "zscore",     # "zscore" | "minmax" | "none"
    metric: str = "euclidean",     # currently only "euclidean" (1D)
    epsilon_mode: str = "quantile",# "quantile" | "fixed" | "none"
    epsilon_q: float = 0.1,        # density target (10%) if epsilon_mode="quantile"
    epsilon_value: float | None = None,  # threshold if epsilon_mode="fixed"
    binarize: bool = True,         # True -> binary RP; False -> continuous distance map
    cmap: str = "gray",
    invert: bool = True            # True -> recurrences are white on black
):
    """
    Render a Recurrence Plot (RP) from a 1D time series (length T).
    Complexity O(T^2); keep windows moderate (e.g., 32..256).

    Steps:
      1) Normalize series (z-score or min-max) to stabilize epsilon selection.
      2) Compute pairwise distances D_ij = |x_i - x_j| (euclidean, 1D).
      3) Choose epsilon:
         - quantile: eps = quantile(D_upper, q)
         - fixed:    eps = epsilon_value
         - none:     no binarization (continuous map)
      4) Build image:
         - binarize: M = 1{D <= eps}
         - else:     M = 1 - D / max(D)  (similarity-like)
      5) Plot with origin="lower", no axes, tight layout.
    """
    x = series.to_numpy(dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("to_recurrence_image expects a 1D series of length >= 2")

    # 1) normalization
    if normalize == "zscore":
        mu, sd = np.nanmean(x), np.nanstd(x)
        x = (x - mu) / (sd + 1e-8)
    elif normalize == "minmax":
        mn, mx = np.nanmin(x), np.nanmax(x)
        x = (x - mn) / (mx - mn + 1e-8)
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be 'zscore', 'minmax', or 'none'")

    # 2) pairwise distance (1D euclidean)
    # D[i,j] = |x[i] - x[j]|
    D = np.abs(x[:, None] - x[None, :])

    # 3) choose epsilon
    eps = None
    if epsilon_mode == "quantile":
        # exclude diagonal zeros to avoid eps=0
        iu = np.triu_indices_from(D, k=1)
        tri = D[iu]
        if len(tri) == 0:
            tri = D.ravel()
        eps = np.quantile(tri, float(epsilon_q))
    elif epsilon_mode == "fixed":
        if epsilon_value is None:
            raise ValueError("epsilon_value must be provided when epsilon_mode='fixed'")
        eps = float(epsilon_value)
    elif epsilon_mode == "none":
        pass
    else:
        raise ValueError("epsilon_mode must be 'quantile', 'fixed', or 'none'")

    # 4) build matrix in [0,1]
    if binarize and eps is not None:
        M = (D <= eps).astype(float)
    else:
        # similarity-like: high = similar (bright), low = dissimilar (dark)
        dmax = np.nanmax(D)
        M = 1.0 - (D / (dmax + 1e-8))

    if not invert:
        M = 1.0 - M

    # 5) plot and save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower", aspect="equal")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()