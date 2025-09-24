#!/usr/bin/env python3
# tools/run_signal_analysis.py
# Analyse du signal entre vols (GK) et y∈{0,1}:
# - Scalar features: Pearson/Spearman/Kendall, ANOVA, Kruskal, MI, AUC stump, Distance Corr, HSIC
# - Multivarié (26 lags GK): LogReg OOF-AUC, MI par lag, Distance Corr multi, HSIC multi
# Sauvegarde les résultats en CSV et imprime un résumé.

from __future__ import annotations
import argparse, sys, math, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Optionnels (on gère leur absence proprement)
try:
    from scipy import stats
except Exception:
    stats = None

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics.pairwise import pairwise_kernels
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
except Exception:
    roc_auc_score = None
    mutual_info_classif = None
    pairwise_kernels = None
    StandardScaler = None
    LogisticRegression = None
    StratifiedKFold = None

try:
    import dcor
except Exception:
    dcor = None


# ---------------------------
# Utilities
# ---------------------------

def ensure_cols(df: pd.DataFrame, cols: List[str], name: str = ""):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Colonnes manquantes pour {name or 'op'}: {miss}")

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # supprime les colonnes Unnamed + remet y en int
    df = df.copy()
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    if "y" in df.columns:
        df["y"] = df["y"].astype(int)
    return df

def choose_scalar_candidates(df: pd.DataFrame, user_feats: List[str] | None) -> List[str]:
    if user_feats:
        feats = [c for c in user_feats if c in df.columns]
        if not feats:
            raise ValueError(f"Aucun des scalar_features demandés n'est présent: {user_feats}")
        return feats
    # défaut: colonnes communes que tu m'as indiquées
    candidates = ["median", "vol_mean", "vol_std", "vol_max", "vol_min"]
    if "gk_tminus1" in df.columns:
        candidates.append("gk_tminus1")
    feats = [c for c in candidates if c in df.columns]
    if not feats:
        raise ValueError("Aucune feature scalaire trouvée (ex: median, vol_mean, ...).")
    return feats

def get_lag_cols(df: pd.DataFrame, max_lag: int = 26) -> List[str]:
    cols = []
    for k in range(1, max_lag+1):
        name = f"gk_tminus{k}"
        if name in df.columns:
            cols.append(name)
    return cols


# ---------------------------
# Scalar metrics
# ---------------------------

def auc_stump(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if roc_auc_score is None:
        return math.nan, math.nan
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    try:
        auc = roc_auc_score(y, x)
    except Exception:
        return math.nan, math.nan
    m1 = np.nanmedian(x[y == 1]) if np.any(y == 1) else math.nan
    m0 = np.nanmedian(x[y == 0]) if np.any(y == 0) else math.nan
    thr = 0.5 * (m0 + m1) if np.isfinite(m0) and np.isfinite(m1) else math.nan
    return float(auc), float(thr)

def distance_corr_scalar(x: np.ndarray, y: np.ndarray) -> float:
    if dcor is None:
        return math.nan
    try:
        return float(dcor.distance_correlation(x, y))
    except Exception:
        return math.nan

def hsic_rbf_scalar(x: np.ndarray, y: np.ndarray, n_perm: int = 0, rng_seed: int = 0) -> Tuple[float, float]:
    if pairwise_kernels is None:
        return math.nan, math.nan
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    n = x.shape[0]
    if n < 5:
        return math.nan, math.nan

    def gamma_median(z: np.ndarray) -> float:
        D = np.abs(np.subtract.outer(z[:, 0], z[:, 0]))
        med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        sigma = med if np.isfinite(med) and med > 0 else 1.0
        return 1.0 / (2.0 * sigma * sigma)

    gx = gamma_median(x)
    gy = gamma_median(y.astype(float))

    K = pairwise_kernels(x, metric="rbf", gamma=gx)
    L = pairwise_kernels(y.astype(float), metric="rbf", gamma=gy)

    H = np.eye(n) - np.ones((n, n)) / n
    KH = H @ K @ H
    LH = H @ L @ H
    hsic = (KH * LH).sum() / ((n - 1) ** 2)

    pval = math.nan
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(rng_seed)
        null = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            Lp = pairwise_kernels(yp.astype(float), metric="rbf", gamma=gy)
            LpH = H @ Lp @ H
            null.append((KH * LpH).sum() / ((n - 1) ** 2))
        null = np.array(null, dtype=float)
        pval = float((np.sum(null >= hsic) + 1) / (len(null) + 1))
    return float(hsic), pval

def compute_scalar_metrics(df: pd.DataFrame, feature: str, y_col: str, n_perm_hsic: int) -> Dict[str, Any]:
    x = df[feature].to_numpy()
    y = df[y_col].to_numpy().astype(int)

    pearson = (math.nan, math.nan)
    spearman_r = spearman_p = math.nan
    kendall_t = kendall_p = math.nan
    f_stat = f_p = kw_stat = kw_p = math.nan
    mi = math.nan
    auc, thr = math.nan, math.nan
    dcor_val = math.nan
    hsic_val, hsic_p = math.nan, math.nan

    if stats is not None:
        try: pearson = stats.pearsonr(x, y)
        except Exception: pass
        try:
            s = stats.spearmanr(x, y, nan_policy="omit")
            spearman_r, spearman_p = float(s.correlation), float(s.pvalue)
        except Exception: pass
        try:
            k = stats.kendalltau(x, y, nan_policy="omit")
            kendall_t, kendall_p = float(k.correlation), float(k.pvalue)
        except Exception: pass
        try:
            x0, x1 = x[y == 0], x[y == 1]
            if len(x0) >= 2 and len(x1) >= 2:
                f_stat, f_p = stats.f_oneway(x0, x1)
                kw_stat, kw_p = stats.kruskal(x0, x1)
        except Exception: pass

    if mutual_info_classif is not None:
        try:
            mi = float(mutual_info_classif(x.reshape(-1, 1), y, discrete_features=False, random_state=0)[0])
        except Exception:
            pass

    auc, thr = auc_stump(x, y)
    dcor_val = distance_corr_scalar(x, y.astype(float))
    hsic_val, hsic_p = hsic_rbf_scalar(x, y, n_perm=n_perm_hsic, rng_seed=0)

    return {
        "feature": feature,
        "n": int(len(df)),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "pearson_r": float(pearson[0]) if np.isfinite(pearson[0]) else math.nan,
        "pearson_p": float(pearson[1]) if np.isfinite(pearson[1]) else math.nan,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
        "kendall_tau": kendall_t,
        "kendall_p": kendall_p,
        "anova_F": float(f_stat) if np.isfinite(f_stat) else math.nan,
        "anova_p": float(f_p) if np.isfinite(f_p) else math.nan,
        "kruskal_H": float(kw_stat) if np.isfinite(kw_stat) else math.nan,
        "kruskal_p": float(kw_p) if np.isfinite(kw_p) else math.nan,
        "mutual_info": mi,
        "auc_stump": auc,
        "thr_stump": thr,
        "dist_corr": dcor_val,
        "hsic_rbf": hsic_val,
        "hsic_p": hsic_p,
    }


# ---------------------------
# Multivariate metrics (26 lags)
# ---------------------------

def oof_auc_logreg(X: np.ndarray, y: np.ndarray, n_splits: int = 5, standardize: bool = True) -> float:
    if any(v is None for v in [LogisticRegression, StratifiedKFold, roc_auc_score]):
        return math.nan
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if X.shape[0] < 10:
        return math.nan

    oof = np.zeros_like(y, dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr, te in cv.split(X, y):
        Xt, Xv = X[tr], X[te]
        yt, yv = y[tr], y[te]
        if standardize and StandardScaler is not None:
            scaler = StandardScaler()
            Xt = scaler.fit_transform(Xt)
            Xv = scaler.transform(Xv)
        model = LogisticRegression(max_iter=2000, n_jobs=None)
        model.fit(Xt, yt)
        oof[te] = model.predict_proba(Xv)[:, 1]
    try:
        return float(roc_auc_score(y, oof))
    except Exception:
        return math.nan

def mi_per_feature(X: np.ndarray, y: np.ndarray, names: List[str]) -> pd.DataFrame:
    if mutual_info_classif is None:
        return pd.DataFrame({"feature": names, "mutual_info": [math.nan]*len(names)})
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if X.shape[0] < 5:
        return pd.DataFrame({"feature": names, "mutual_info": [math.nan]*len(names)})
    vals = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    return pd.DataFrame({"feature": names, "mutual_info": vals})

def distance_corr_multi(X: np.ndarray, y: np.ndarray) -> float:
    if dcor is None:
        return math.nan
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if X.shape[0] < 5:
        return math.nan
    try:
        return float(dcor.distance_correlation(X, y.astype(float)))
    except Exception:
        return math.nan

def hsic_rbf_multi(X: np.ndarray, y: np.ndarray, n_perm: int = 0, rng_seed: int = 0) -> Tuple[float, float]:
    if pairwise_kernels is None:
        return math.nan, math.nan
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    n = X.shape[0]
    if n < 5:
        return math.nan, math.nan

    def gamma_median_matrix(Z: np.ndarray) -> float:
        D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
        med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        sigma = med if np.isfinite(med) and med > 0 else 1.0
        return 1.0 / (2.0 * sigma * sigma)

    gx = gamma_median_matrix(X)
    y2 = y.reshape(-1, 1).astype(float)
    gy = gamma_median_matrix(y2)

    K = pairwise_kernels(X, metric="rbf", gamma=gx)
    L = pairwise_kernels(y2, metric="rbf", gamma=gy)

    H = np.eye(n) - np.ones((n, n)) / n
    KH = H @ K @ H
    LH = H @ L @ H
    hsic = (KH * LH).sum() / ((n - 1) ** 2)

    pval = math.nan
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(rng_seed)
        null = []
        for _ in range(n_perm):
            yp = rng.permutation(y2)
            Lp = pairwise_kernels(yp, metric="rbf", gamma=gy)
            LpH = H @ Lp @ H
            null.append((KH * LpH).sum() / ((n - 1) ** 2))
        null = np.array(null, dtype=float)
        pval = float((np.sum(null >= hsic) + 1) / (len(null) + 1))
    return float(hsic), pval


# ---------------------------
# Main
# ---------------------------

def run(args):
    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    df = clean_df(df)
    ensure_cols(df, ["y"], "label y")
    # groupes (facultatifs)
    group_keys: List[str] = []
    if args.group_by:
        for g in args.group_by.split(","):
            g = g.strip()
            if g:
                if g not in df.columns:
                    warnings.warn(f"[WARN] group_by colonne absente: {g}")
                else:
                    group_keys.append(g)
    if not group_keys:
        groups = [("ALL", df)]
    else:
        groups = list(df.groupby(group_keys, dropna=False))
        # format propre du nom de groupe
        groups = [("_".join(f"{k}={v}" for k, v in zip(group_keys, (g if isinstance(g, tuple) else (g,)))), gdf) for g, gdf in groups]

    # scalar features
    scalar_feats = choose_scalar_candidates(df, args.scalar_features.split(",") if args.scalar_features else None)

    # lag matrix (multivarié)
    lag_cols = get_lag_cols(df, max_lag=args.max_lag)
    X_all = df[lag_cols].to_numpy() if lag_cols else None

    scalar_rows: List[Dict[str, Any]] = []
    multi_rows: List[Dict[str, Any]] = []

    for gname, gdf in groups:
        # filtrage NaN/inf
        gdf = gdf.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"], how="any")
        y = gdf["y"].to_numpy().astype(int)
        if len(gdf) < 5 or np.unique(y).size < 2:
            print(f"[SKIP] Group={gname}: n={len(gdf)} or not enough class variety.")
            continue

        print("=" * 88)
        print(f"[Group={gname}] n={len(gdf)} | positives={(y==1).sum()} negatives={(y==0).sum()}")

        # --- Scalar metrics for each candidate ---
        # for feat in scalar_feats:
        #     if feat not in gdf.columns:
        #         continue
        #     gdf_feat = gdf.dropna(subset=[feat])
        #     if gdf_feat.empty:
        #         continue
        #     row = {"group": gname}
        #     row.update(compute_scalar_metrics(gdf_feat, feat, "y", n_perm_hsic=args.hsic_permutations))
        #     scalar_rows.append(row)
        #     # short print
        #     print(f"  • {feat:<12} | AUC(stump)={row['auc_stump']:.3f}  MI={row['mutual_info']:.4f} "
        #           f"Pearson r={row['pearson_r']:.3f} (p={row['pearson_p']:.1e})  dCor={row['dist_corr']:.3f}")

        # --- Multivariate (26 lags) ---
        if lag_cols:
            Xg = gdf[lag_cols].to_numpy()
            yg = gdf["y"].to_numpy().astype(int)

            auc_oof = oof_auc_logreg(Xg, yg, n_splits=args.cv, standardize=True)
            dcor_m  = distance_corr_multi(Xg, yg)
            hsic_m, hsic_p = hsic_rbf_multi(Xg, yg, n_perm=args.hsic_permutations)

            mi_df = mi_per_feature(Xg, yg, lag_cols)
            mi_df.insert(0, "group", gname)
            # top-5 lags by MI (print)
            if not mi_df["mutual_info"].isna().all():
                top5 = mi_df.sort_values("mutual_info", ascending=False).head(5)
                print("  • Top-5 lags by MI:", ", ".join(f"{r.feature}:{r.mutual_info:.4f}" for r in top5.itertuples()))

            multi_rows.append({
                "group": gname,
                "n": int(len(gdf)),
                "auc_logreg_oof": float(auc_oof) if np.isfinite(auc_oof) else math.nan,
                "dist_corr_multi": float(dcor_m) if np.isfinite(dcor_m) else math.nan,
                "hsic_rbf_multi": float(hsic_m) if np.isfinite(hsic_m) else math.nan,
                "hsic_rbf_multi_p": float(hsic_p) if np.isfinite(hsic_p) else math.nan,
            })

            # Save MI per lag per group
            mi_path = out_dir / f"mi_per_lag__{sanitize(gname)}.csv"
            mi_df.to_csv(mi_path, index=False)
        else:
            print("  • No lag columns found → multivariate block skipped.")

    # --- Save scalar & multivariate summaries ---
    if scalar_rows:
        pd.DataFrame(scalar_rows).to_csv(out_dir / "scalar_metrics.csv", index=False)
    if multi_rows:
        pd.DataFrame(multi_rows).to_csv(out_dir / "multivariate_summary.csv", index=False)

    print("\n[OK] Résultats sauvegardés dans:", out_dir.resolve())
    if scalar_rows:
        print("   - scalar_metrics.csv")
    if multi_rows:
        print("   - multivariate_summary.csv")
        print("   - mi_per_lag__<group>.csv")

def sanitize(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_=+." else "_" for ch in str(s))


# ---------------------------
# CLI
# ---------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Signal analysis (vol GK vs y).")
    p.add_argument("--csv", required=True, help="Chemin du CSV d'entrée.")
    p.add_argument("--out_dir", default="signal_out", help="Dossier de sortie des rapports CSV.")
    p.add_argument("--group_by", default="split,symbol", help="Colonnes de groupement (ex: 'split,symbol' ou vide).")
    p.add_argument("--scalar_features", default="", help="Liste de features scalaires à tester (séparées par des virgules). Vide = auto.")
    p.add_argument("--max_lag", type=int, default=26, help="Nombre max de lags GK à chercher (gk_tminus1..max_lag).")
    p.add_argument("--cv", type=int, default=5, help="Nombre de folds pour OOF-AUC LogReg.")
    p.add_argument("--hsic_permutations", type=int, default=0, help="Nb de permutations pour p-val HSIC (0 = skip).")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
