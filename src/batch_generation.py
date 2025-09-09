#!/usr/bin/env python3
# src/batch_generation.py
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import argparse

# module frère (exécuter avec: python -m JPMC_CNN_vol_project.src.batch_generation)
from .data_generation import (
    build_dataset,
    load_yaml_config,
    merge_args_with_config,
)

def parse_args():
    p = argparse.ArgumentParser(description="Batch: generate dataset for many CSVs using one YAML.")
    p.add_argument("--csv_dir", required=True, help="Folder containing per-symbol CSV files.")
    p.add_argument("--config", required=True, help="Path to dataset.yaml (global defaults).")
    p.add_argument("--pattern", default="*.csv", help="Glob pattern (default: *.csv).")
    return p.parse_args()

def main():
    a = parse_args()
    csv_dir = Path(a.csv_dir).resolve()
    cfg_path = Path(a.config).resolve()
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"csv_dir not found: {csv_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    # Charge le YAML (utile si tu veux valider/echo, merge_args_with_config le relira aussi)
    _ = load_yaml_config(str(cfg_path))

    files = sorted(csv_dir.glob(a.pattern))
    if not files:
        raise RuntimeError(f"No CSV matching '{a.pattern}' in {csv_dir}")

    for csv_path in files:
        symbol = csv_path.stem  # ex: AAPL pour AAPL.csv
        print(f"[RUN] sym={symbol} csv={csv_path}")

        # 1) Args minimaux pour surcharger le YAML
        args = SimpleNamespace(
            # overrides
            config=str(cfg_path),
            csv=str(csv_path),
            symbol=symbol,
            out=None,  # None -> prendra output.dir du YAML

            # --- tout le reste à None : le YAML fournit les valeurs ---
            # data / resample
            time_col=None, symbol_col=None, symbols=None, tz_utc=None,
            rs_rule=None, rs_label=None, rs_closed=None, rs_dropna=None,

            # splits
            train_end=None, val_end=None,

            # labels
            horizons=None, median_window=None,

            # deseason
            deseason_mode=None, deseason_bucket=None,

            # images (général)
            image_encoder=None, image_windows=None, image_step=None,
            ts_ma_window=None, img_w=None, img_h=None, img_dpi=None,

            # heatmap
            heatmap_mode=None, heatmap_cmap=None, heatmap_vmin=None, heatmap_vmax=None,

            # evaluation
            embargo_steps=None, no_overlap=None,

            # alignment (ancre journalière)
            align_enabled=None, align_time_str=None, align_tz=None,

            # day separators
            day_sep_enabled=None, day_sep_label=None, day_sep_color=None,
            day_sep_alpha=None, day_sep_lw=None, day_label_kind=None, day_label_lang=None,

            # TS overlays
            show_ma_top=None, bb_enabled=None, bb_window=None, bb_nstd=None,
            bottom_panel=None, rsi_window=None, rsi_source=None,

            # couleurs
            fg=None, bg=None,

            # Recurrence Plot
            rp_series=None, rp_normalize=None, rp_metric=None,
            rp_epsilon_mode=None, rp_epsilon_q=None, rp_epsilon_value=None,
            rp_binarize=None, rp_cmap=None, rp_invert=None,

            # Gramian Angular Fields
            gaf_mode=None, gaf_normalize=None, gaf_cmap=None, gaf_invert=None,

            # nettoyage éventuel
            clean=None, clean_scope=None,
        )

        # 2) Merge YAML + overrides -> m
        m = merge_args_with_config(args)
        splits = (m.train_end, m.val_end) if m.train_end else None

        # 3) Appel pipeline classique avec TOUS les paramètres
        build_dataset(
            csv_path=Path(m.csv),
            out_dir=Path(m.out),
            symbol=m.symbol,

            # labels
            horizon_list=m.horizons,
            median_window=m.median_window,

            # images
            image_windows=m.image_windows,
            image_step=m.image_step,
            image_encoder=m.image_encoder,
            ts_ma_window=m.ts_ma_window,
            img_w=m.img_w, img_h=m.img_h, img_dpi=m.img_dpi,

            # splits & deseason
            splits=splits,
            deseason_mode=m.deseason_mode,
            deseason_bucket=m.deseason_bucket,

            # RESAMPLE (nouveau)
            rs_rule=m.rs_rule, rs_label=m.rs_label, rs_closed=m.rs_closed, rs_dropna=m.rs_dropna,

            # alignment
            align_enabled=m.align_enabled,
            align_time_str=m.align_time_str,
            align_tz=m.align_tz,

            # heatmap
            heatmap_cmap=m.heatmap_cmap,
            heatmap_vmin=m.heatmap_vmin,
            heatmap_vmax=m.heatmap_vmax,

            # day separators
            day_sep_enabled=m.day_sep_enabled,
            day_sep_label=m.day_sep_label,
            day_sep_color=m.day_sep_color,
            day_sep_alpha=m.day_sep_alpha,
            day_sep_lw=m.day_sep_lw,
            day_label_kind=m.day_label_kind,

            # TS overlays
            show_ma_top=m.show_ma_top,
            bb_enabled=m.bb_enabled,
            bb_window=m.bb_window,
            bb_nstd=m.bb_nstd,
            bottom_panel=m.bottom_panel,
            rsi_window=m.rsi_window,
            rsi_source=m.rsi_source,

            # couleurs
            fg=m.fg, bg=m.bg,

            # Recurrence Plot
            rp_series=m.rp_series,
            rp_normalize=m.rp_normalize,
            rp_metric=m.rp_metric,
            rp_epsilon_mode=m.rp_epsilon_mode,
            rp_epsilon_q=m.rp_epsilon_q,
            rp_epsilon_value=m.rp_epsilon_value,
            rp_binarize=m.rp_binarize,
            rp_cmap=m.rp_cmap,
            rp_invert=m.rp_invert,

            # Gramian Angular Fields
            gaf_mode=m.gaf_mode,
            gaf_normalize=m.gaf_normalize,
            gaf_cmap=m.gaf_cmap,
            gaf_invert=m.gaf_invert,
            out_path_batch='dataset_out_ts/all_ticker'
        )

    print("[DONE] all symbols processed.")

if __name__ == "__main__":
    main()