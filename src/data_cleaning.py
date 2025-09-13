import pandas as pd
import numpy as np
from typing import Iterable, Tuple, List, Optional

def _ensure_ts_utc_floor_minute(s: pd.Series) -> pd.Series:
    """
    Parse -> tz-aware UTC -> floor to minute.

    - utc=True :
        * si timestamps naïfs → considérés comme UTC
        * s’ils ont un fuseau → convertis vers UTC
    """
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts.dt.floor("T")

def _constant_fill(value_series: pd.Series):
    """Première valeur non nulle (utile pour méta constantes par symbole)."""
    if value_series.notna().any():
        return value_series.dropna().iloc[0]
    return np.nan

def interpolate_minute_ohlcv(
    df: pd.DataFrame,
    ts_col: str = "ts_event",
    price_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
    volume_col: str = "volume",
    group_cols: Iterable[str] = ("symbol",),   # ajoutez "instrument_id" si nécessaire
    tz_market: str = "America/New_York",
    market_open: str = "09:30",
    market_close: str = "16:00",
    keep_columns_as_constant: Optional[Iterable[str]] = ("rtype", "publisher_id", "instrument_id", "symbol"),
) -> pd.DataFrame:
    """
    1) Complète la grille minute par minute pour chaque groupe (symbol, …).
    2) Crée des barres synthétiques pour les minutes manquantes (voir règles ci-dessous).
    3) Filtre les minutes RTH (lun-ven, 09:30–16:00 NY).
    Retourne un DataFrame indexé par horodatage (UTC) + colonne booléenne 'was_interpolated'.

    Politique par défaut sur minutes manquantes:
      close = ffill(close) ; open = high = low = close synthétique ; volume = 0
    """
    required = set(price_cols + (volume_col,))
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    # Validation des colonnes de groupe
    missing_group = set(group_cols) - set(df.columns)
    if missing_group:
        raise ValueError(f"Missing group columns: {missing_group}")

    # 0) Nettoyage time index
    df = df.copy()
    df[ts_col] = _ensure_ts_utc_floor_minute(df[ts_col])

    # On élimine les timestamps invalides
    df = df.dropna(subset=[ts_col])
    df = df.sort_values([*group_cols, ts_col])

    # 1) Traitement par groupe
    out_parts: List[pd.DataFrame] = []
    for keys, g in df.groupby(list(group_cols), sort=False):
        g = g.copy().set_index(ts_col)

        # Dédupliquer si besoin (garde la dernière ligne par minute)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) == 0:
            continue

        # Grille minute complète (UTC) du min au max
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="T", tz="UTC")

        # Reindex pour matérialiser les trous
        g_re = g.reindex(full_idx)

        # Marqueur des lignes synthétiques
        was_missing = g_re[price_cols[-1]].isna()  # teste 'close' NaN

        # --- Règle de remplissage ---
        close_ffill = g_re[price_cols[-1]].ffill()
        open_new  = g_re[price_cols[0]].where(~was_missing, close_ffill)
        high_new  = g_re[price_cols[1]].where(~was_missing, close_ffill)
        low_new   = g_re[price_cols[2]].where(~was_missing, close_ffill)
        close_new = close_ffill
        volume_new = g_re[volume_col].where(~was_missing, 0)

        filled = pd.DataFrame({
            price_cols[0]: open_new,
            price_cols[1]: high_new,
            price_cols[2]: low_new,
            price_cols[3]: close_new,
            volume_col: volume_new,
        }, index=full_idx)

        # Remplir les colonnes constantes UNIQUEMENT si elles existent
        if keep_columns_as_constant:
            available_const = tuple(c for c in keep_columns_as_constant if c in df.columns)
            for c in available_const:
                const_val = _constant_fill(g_re[c]) if c in g_re.columns else _constant_fill(df[c])
                filled[c] = const_val  # affectation directe (constante)

        # Ajouter les autres colonnes (ni OHLCV ni constantes) avec ffill/bfill
        const_set = set(keep_columns_as_constant or [])
        other_cols = [c for c in g_re.columns if c not in set(price_cols + (volume_col,)) | const_set]
        for c in other_cols:
            filled[c] = g_re[c].ffill().bfill()

        filled["was_interpolated"] = was_missing.fillna(False).astype(bool)

        # Tag des clés de groupe
        if isinstance(keys, tuple):
            for name, val in zip(group_cols, keys):
                filled[name] = val
        else:
            filled[next(iter(group_cols))] = keys

        out_parts.append(filled)

    if not out_parts:
        return pd.DataFrame(
            columns=[*group_cols, *price_cols, volume_col, "was_interpolated"]
        ).set_index(pd.DatetimeIndex([], tz="UTC"))

    full = pd.concat(out_parts, axis=0)
    full.index.name = ts_col  # index = UTC minute grid

    # 2) Filtre horaires RTH (New York)
    idx_ny = full.index.tz_convert(tz_market)
    is_weekday = idx_ny.dayofweek < 5
    t_open = pd.to_datetime(market_open).time()
    t_close = pd.to_datetime(market_close).time()
    tod = idx_ny.time
    in_rth = (tod >= t_open) & (tod < t_close)

    full_rth = full[is_weekday & in_rth].copy()

    # Tri final pour propreté
    full_rth = full_rth.sort_values(list(group_cols) + [ts_col])

    return full_rth