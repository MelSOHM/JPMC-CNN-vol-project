# indicators.py
# Lightweight, vectorized technical indicators (no look-ahead).
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import shutil, os, stat

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

# --- safe cleanup utilities ---
def _chmod_rw(path: Path):
    """Ensure path is deletable. Directories need +x to be traversable."""
    try:
        if path.is_dir():
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)   # 0o700 on owner
        else:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)                   # 0o600 on owner
    except Exception:
        pass

def _onerror_chmod(func, path, exc_info):
    """
    rmtree onerror handler: make the path writable/executable (if dir) then retry.
    """
    try:
        p = Path(path)
        _chmod_rw(p)
        func(path)  # retry the failed operation (unlink/rmdir)
    except Exception:
        # give up; shutil will raise after this handler returns
        pass

def clean_dataset_dir(out_dir: Path,
                      *,
                      symbol: str | None = None,
                      what: str = "symbol") -> int:
    """
    Remove previously generated artifacts before a new run.

    what:
      - "symbol": delete <out_dir>/<symbol>
      - "splits": delete only train/val/test/full under <out_dir>/<symbol>
      - "all":    delete all children under <out_dir>
    """
    out_dir = Path(out_dir).resolve()
    if not out_dir.exists():
        return 0
    if str(out_dir) in ("/", "", str(Path.home())):
        raise RuntimeError(f"Refuse to clean unsafe path: {out_dir}")

    def _rmtree(p: Path):
        if not p.exists():
            return
        if p.is_symlink():
            # don't descend into symlinks; just unlink
            p.unlink(missing_ok=True)
            return
        # pre-pass: best effort to make everything deletable
        for root, dirs, files in os.walk(p, topdown=False, followlinks=False):
            for name in files:
                _chmod_rw(Path(root) / name)
            for name in dirs:
                _chmod_rw(Path(root) / name)
        # main removal with robust onerror handler
        shutil.rmtree(p, ignore_errors=False, onerror=_onerror_chmod)

    removed = 0
    if what == "all":
        for child in out_dir.iterdir():
            _rmtree(child); removed += 1
    elif what == "symbol":
        if not symbol:
            raise ValueError("clean_dataset_dir: 'symbol' is required when what='symbol'.")
        target = out_dir / symbol
        if target.exists():
            _rmtree(target); removed += 1
    elif what == "splits":
        if not symbol:
            raise ValueError("clean_dataset_dir: 'symbol' is required when what='splits'.")
        for split in ("train", "val", "test", "full"):
            target = out_dir / symbol / split
            if target.exists():
                _rmtree(target); removed += 1
    else:
        raise ValueError("clean_dataset_dir: 'what' must be one of {'symbol','splits','all'}.")

    return removed