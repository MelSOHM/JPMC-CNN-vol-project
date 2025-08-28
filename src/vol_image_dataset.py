# src/vol_image_dataset.py
# Volatility image dataset for PyTorch: scans generated folders and yields (image, label[, meta])
# Compatible with folders: <root>/<symbol>/<split>/d{N}/y{0|1}/*.png
# Backward-compatible with older naming: h{N}/y{0|1}/*.png

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from PIL import Image
from torchvision import transforms as T


# -----------------------------
# Small structures and utilities
# -----------------------------

@dataclass(frozen=True)
class VolImageSample:
    """Lightweight record describing one sample on disk."""
    image_path: Path
    label: int                 # 0 or 1
    split: str                 # 'train' | 'val' | 'test' | 'full'
    symbol: str                # e.g. 'AAPL'
    horizon_days: int          # extracted from 'd{N}' or 'h{N}' folder
    timestamp_str: str         # parsed from filename if available, else basename


TIMESTAMP_STEM_RE = re.compile(r"(\d{8}T\d{6}\d{0,6}Z)")

def _parse_timestamp_stem(p: Path) -> str:
    """
    Try to extract a timestamp-like stem produced by ts_to_filename (e.g. 20250820T143000123456Z).
    Fall back to the filename stem if no match is found.
    """
    m = TIMESTAMP_STEM_RE.search(p.stem)
    return m.group(1) if m else p.stem


def _infer_split_roots(root: Path, symbols: Optional[Sequence[str]]) -> List[Tuple[str, Path]]:
    """
    Enumerate (symbol, path) roots that contain split subfolders.
    If `symbols` is None, use all top-level directories.
    """
    roots: List[Tuple[str, Path]] = []
    if symbols:
        for sym in symbols:
            p = root / sym
            if p.is_dir():
                roots.append((sym, p))
    else:
        for p in root.iterdir():
            if p.is_dir():
                roots.append((p.name, p))
    return roots


def _horizon_from_dirname(name: str) -> Optional[int]:
    """
    Extract horizon from a directory name:
      - new: d{N} (days)
      - legacy: h{N} (bars)
    Returns None if not matching.
    """
    if len(name) >= 2 and (name[0] in ("d", "h")) and name[1:].isdigit():
        return int(name[1:])
    return None


def _scan_split_dir(
    split_dir: Path,
    split_name: str,
    symbol: str,
    horizons_filter: Optional[Sequence[int]] = None,
    prefer_days: bool = True,
) -> List[VolImageSample]:
    """
    Scan a split directory and collect samples.
    We support both d{N}/y*/ and h{N}/y*/ layouts. If both exist, prefer d{N} when `prefer_days=True`.
    """
    out: List[VolImageSample] = []

    # Find horizon folders
    horizon_dirs: List[Tuple[int, Path]] = []
    for child in split_dir.iterdir():
        if not child.is_dir():
            continue
        N = _horizon_from_dirname(child.name)
        if N is None:
            continue
        # If both dN and hN exist, prefer dN when requested
        if prefer_days and child.name.startswith("h"):
            # skip 'hN' if 'dN' exists
            d_alt = split_dir / f"d{N}"
            if d_alt.exists():
                continue
        horizon_dirs.append((N, child))

    if horizons_filter:
        horizon_dirs = [(N, p) for (N, p) in horizon_dirs if N in set(horizons_filter)]

    for horizon_days, hdir in horizon_dirs:
        # Class subfolders
        for y in (0, 1):
            cls_dir = hdir / f"y{y}"
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.glob("*.png"):
                out.append(
                    VolImageSample(
                        image_path=img_path,
                        label=int(y),
                        split=split_name,
                        symbol=symbol,
                        horizon_days=horizon_days,
                        timestamp_str=_parse_timestamp_stem(img_path),
                    )
                )
    return out


def scan_dataset(
    root: Path,
    splits: Sequence[str] = ("train", "val", "test"),
    symbols: Optional[Sequence[str]] = None,
    horizons: Optional[Sequence[int]] = None,
    prefer_days: bool = True,
) -> List[VolImageSample]:
    """
    Discover all image samples under `root`.
    Expected structure:
      <root>/<symbol>/<split>/(d{N}|h{N})/y{0|1}/*.png
    """
    samples: List[VolImageSample] = []
    for symbol, sym_root in _infer_split_roots(root, symbols):
        for split_name in splits:
            split_dir = sym_root / split_name
            if not split_dir.is_dir():
                continue
            samples.extend(
                _scan_split_dir(split_dir, split_name, symbol, horizons_filter=horizons, prefer_days=prefer_days)
            )
    return samples


def compute_class_weights(samples: Sequence[VolImageSample]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for BCE/CE losses.
    weight[c] = N_total / (num_classes * N_c)
    """
    if not samples:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    counts = {0: 0, 1: 0}
    for s in samples:
        counts[s.label] += 1
    n0, n1 = counts[0], counts[1]
    total = n0 + n1
    # avoid div-by-zero
    w0 = (total / (2.0 * max(1, n0)))
    w1 = (total / (2.0 * max(1, n1)))
    return torch.tensor([w0, w1], dtype=torch.float32)


# ----------------------
# Core Dataset class
# ----------------------

class VolImageDataset(Dataset):
    """
    Generic image dataset for the volatility CNN project.

    Each item is:
      image: Tensor [C,H,W] in float32 (after transforms)
      label: int (0 or 1)
      meta:  dict with { "split", "symbol", "horizon_days", "timestamp", "path" }  (if return_meta=True)
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        symbols: Optional[Sequence[str]] = None,
        horizons: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
        prefer_days: bool = True,
        force_rgb: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_meta = return_meta
        self.force_rgb = force_rgb

        all_samples = scan_dataset(
            self.root,
            splits=(split,),
            symbols=symbols,
            horizons=horizons,
            prefer_days=prefer_days,
        )
        if not all_samples:
            raise RuntimeError(
                f"No samples found under {self.root} for split='{split}'. "
                f"Expected <root>/<symbol>/{split}/dN(or hN)/y0|y1/*.png"
            )
        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path)

        # Convert mode: many heatmaps/ts_vol are saved as RGB; some may be L (grayscale).
        if self.force_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = int(s.label)

        if self.return_meta:
            meta = {
                "split": s.split,
                "symbol": s.symbol,
                "horizon_days": s.horizon_days,
                "timestamp": s.timestamp_str,
                "path": str(s.image_path),
            }
            return img, label, meta
        return img, label


# ----------------------
# Transforms and loaders
# ----------------------

def build_transforms(
    img_size: Tuple[int, int] = (256, 256),
    normalize: Optional[str] = "imagenet",
    train_augment: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Build (train_tfms, eval_tfms).
    We avoid horizontal flips (time axis would be reversed). Light intensity jitter is allowed.
    """
    if normalize == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif normalize is None:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError("normalize must be 'imagenet' or None")

    common = [T.Resize(img_size), T.ToTensor(), T.Normalize(mean, std)]

    if train_augment:
        # Gentle augmentations that preserve temporal structure
        train_tfms = T.Compose([
            T.Resize(img_size),
            # small brightness/contrast jitter (no hue/sat change for BW-ish plots)
            T.ColorJitter(brightness=0.1, contrast=0.1),
            # tiny horizontal translation is OK; avoid flips/rotations that break time axis
            # If you prefer zero translation, comment the next line:
            T.RandomAffine(degrees=0, translate=(0.02, 0.0)),  # up to 2% width shift
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        train_tfms = T.Compose(common)

    eval_tfms = T.Compose(common)
    return train_tfms, eval_tfms


def make_dataloaders(
    root: Path | str,
    symbols: Optional[Sequence[str]] = None,
    horizons: Optional[Sequence[int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (256, 256),
    normalize: Optional[str] = "imagenet",
    train_augment: bool = False,
    prefer_days: bool = True,
    return_meta: bool = False,
    balanced_sampler: bool = False,
) -> Dict[str, DataLoader]:
    """
    Convenience factory returning dataloaders for train/val/test (if present on disk).

    If balanced_sampler=True, the train loader uses a WeightedRandomSampler computed from class frequencies.
    """
    train_tfms, eval_tfms = build_transforms(img_size=img_size, normalize=normalize, train_augment=train_augment)

    loaders: Dict[str, DataLoader] = {}
    splits_to_try = ("train", "val", "test")

    # Build datasets that exist on disk
    datasets: Dict[str, VolImageDataset] = {}
    for split in splits_to_try:
        try:
            ds = VolImageDataset(
                root=root,
                split=split,
                symbols=symbols,
                horizons=horizons,
                transform=(train_tfms if split == "train" else eval_tfms),
                return_meta=return_meta,
                prefer_days=prefer_days,
                force_rgb=True,
            )
            datasets[split] = ds
        except RuntimeError:
            # split missing on disk; skip silently
            pass

    # Train loader (with optional class balancing)
    if "train" in datasets:
        train_ds = datasets["train"]
        if balanced_sampler:
            # Compute per-class weights → per-sample weights → sampler
            w = compute_class_weights(train_ds.samples)  # [w0, w1]
            sample_w = [w[s.label].item() for s in train_ds.samples]
            sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                      num_workers=num_workers, pin_memory=True, drop_last=False)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=False)
        loaders["train"] = train_loader

    # Eval loaders (no shuffle)
    for split in ("val", "test"):
        if split in datasets:
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )

    if not loaders:
        raise RuntimeError(f"No split folders found under {root}. Nothing to load.")
    return loaders
