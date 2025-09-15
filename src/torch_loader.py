# src/torch_loader.py
# Build PyTorch DataLoaders from your YAML config for image-only training.
# Directory layout expected:
#   <root>/<symbol>/<split>/(d{N}|h{N})/y{0|1}/*.png
#
# No auxiliary metadata/features are used here: only the image pixels and class
# (parsed from folder y0/y1). Perfect for "images-only" experiments.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import re
import warnings
import argparse

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms as T
from PIL import Image

# ---- YAML loader ----
try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e


# ---------------------------
# Tiny config helpers
# ---------------------------

def _get(d: dict, path: str, default=None):
    """Dot-path get with defaults (e.g., _get(cfg, 'images.size.width', 256))."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------
# Dataset scanning & sample
# ---------------------------

@dataclass(frozen=True)
class ImgSample:
    path: Path
    label: int                 # 0 or 1 (from folder y0/y1)
    split: str                 # 'train' | 'val' | 'test' | 'full'
    symbol: str
    horizon: int               # from d{N} or h{N}
    timestamp: str             # extracted from filename stem (best effort)


TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6}\d{0,6}Z)")

def _parse_ts(p: Path) -> str:
    m = TIMESTAMP_RE.search(p.stem)
    return m.group(1) if m else p.stem

def _horizon_from_dirname(name: str) -> Optional[int]:
    if len(name) >= 2 and name[0] in ("d", "h") and name[1:].isdigit():
        return int(name[1:])
    return None

def _infer_symbols(root: Path) -> List[str]:
    syms = [p.name for p in root.iterdir() if p.is_dir()]
    if not syms:
        raise RuntimeError(f"No symbols found under {root}.")
    return sorted(syms)

def _scan_split(sym_root: Path,
                split: str,
                symbol: str,
                horizons: Optional[Sequence[int]],
                prefer_days: bool) -> List[ImgSample]:
    out: List[ImgSample] = []
    split_dir = sym_root / split
    if not split_dir.is_dir():
        return out

    # list candidate horizon folders
    cand: List[Tuple[int, Path]] = []
    for child in split_dir.iterdir():
        if not child.is_dir():
            continue
        N = _horizon_from_dirname(child.name)
        if N is None:
            continue
        # prefer dN over hN if both exist
        if prefer_days and child.name.startswith("h") and (split_dir / f"d{N}").exists():
            continue
        if horizons is not None and N not in horizons:
            continue
        cand.append((N, child))

    for N, hdir in sorted(cand):
        for y in (0, 1):
            cls = hdir / f"y{y}"
            if not cls.is_dir():
                continue
            for img_path in cls.glob("*.png"):
                out.append(ImgSample(
                    path=img_path,
                    label=int(y),
                    split=split,
                    symbol=symbol,
                    horizon=N,
                    timestamp=_parse_ts(img_path),
                ))
    return out

def scan_images(root: Path,
                symbols: Optional[Sequence[str]],
                splits: Sequence[str],
                horizons: Optional[Sequence[int]],
                prefer_days: bool) -> List[ImgSample]:
    """Discover all image samples under root."""
    if symbols is None:
        symbols = _infer_symbols(root)
    samples: List[ImgSample] = []
    for sym in symbols:
        sym_root = root / sym if sym != 'batch' else root
        if not sym_root.is_dir():
            continue
        for split in splits:
            samples.extend(_scan_split(sym_root, split, sym, horizons, prefer_days))
    return samples


# ---------------------------
# Core PyTorch dataset
# ---------------------------

class ImageOnlyDataset(Dataset):
    """
    Returns (image_tensor, label_int) or (image_tensor, label_int, meta_dict).

    The dataset reads the on-disk structure and *does not* use any auxiliary
    metadata CSV (pixels-only path).
    """

    def __init__(self,
                 samples: List[ImgSample],
                 transform: Optional[object] = None,
                 return_meta: bool = False,
                 force_rgb: bool = True):
        super().__init__()
        if not samples:
            raise RuntimeError("Empty sample list for this split.")
        self.samples = samples
        self.transform = transform
        self.return_meta = return_meta
        self.force_rgb = force_rgb

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path)
        if self.force_rgb:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = int(s.label)
        if self.return_meta:
            meta = {"path": str(s.path), "symbol": s.symbol, "split": s.split,
                    "horizon": s.horizon, "timestamp": s.timestamp}
            return img, y, meta
        return img, y


# ---------------------------
# Transforms & weights
# ---------------------------

def build_transforms(img_size: Tuple[int, int],
                     normalize: Optional[str] = "imagenet",
                     train_augment: bool = False) -> Tuple[object, object]:
    """Build (train_tfms, eval_tfms). Avoid flips/rotations (time axis)."""
    if normalize == "imagenet":
        mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    elif normalize is None:
        mean = (0.5, 0.5, 0.5); std = (0.5, 0.5, 0.5)
    else:
        raise ValueError("training.normalize must be 'imagenet' or null")

    eval_tfms = T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean, std)])
    if train_augment:
        train_tfms = T.Compose([
            T.Resize(img_size),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.RandomAffine(degrees=0, translate=(0.02, 0.0)),  # small horizontal shift
            T.ToTensor(), T.Normalize(mean, std)
        ])
    else:
        train_tfms = eval_tfms
    return train_tfms, eval_tfms

def compute_class_weights(samples: Sequence[ImgSample]) -> torch.Tensor:
    """Inverse frequency weights: w_c = N / (2 * N_c)."""
    counts = {0: 0, 1: 0}
    for s in samples: counts[s.label] += 1
    total = counts[0] + counts[1]
    w0 = total / (2.0 * max(1, counts[0])); w1 = total / (2.0 * max(1, counts[1]))
    return torch.tensor([w0, w1], dtype=torch.float32)


# ---------------------------
# Loader factory (YAML-driven)
# ---------------------------

def make_dataloaders_from_yaml(cfg_path: Union[str, Path]) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from config YAML.
    Only pixels (images) are used; labels come from y0/y1 folders.
    """
    cfg_path = Path(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    batch = _get(cfg, "output.batch_dir", None)
    if batch:
        root = Path(batch)
    else:
        root = Path(_get(cfg, "output.dir", "./dataset_out")).resolve()
    if not root.exists():
        raise FileNotFoundError(f"output.dir not found: {root}")

    # symbols: prefer explicit array, then single data.symbol, else infer
    symbols = _get(cfg, "data.symbols", None)
    if symbols is None:
        sym = _get(cfg, "data.symbol", None)
        symbols = [sym] if sym else None

    # horizons filter (optional)
    horizons = _get(cfg, "labels.horizons", None)
    if horizons is not None:
        horizons = [int(h) for h in horizons]

    # training params
    bs          = int(_get(cfg, "training.batch_size", 64))
    nw          = int(_get(cfg, "training.num_workers", 4))
    normalize   = _get(cfg, "training.normalize", "imagenet")  # 'imagenet' or null
    augment     = bool(_get(cfg, "training.augment", False))
    prefer_days = bool(_get(cfg, "training.prefer_days", True))
    return_meta = bool(_get(cfg, "training.return_meta", True))
    balanced    = bool(_get(cfg, "training.balanced_sampler", False))

    # image size: training.img_size overrides images.size.<w/h> if present
    img_w = _get(cfg, "training.img_size.0", None)
    img_h = _get(cfg, "training.img_size.1", None)
    if img_w is None or img_h is None:
        img_w = int(_get(cfg, "images.size.width", 256))
        img_h = int(_get(cfg, "images.size.height", 256))
    img_size = (int(img_w), int(img_h))

    # scan disk
    if symbols:
        all_samples = scan_images(root, symbols, ("train", "val", "test"), horizons, prefer_days)
    else:
        all_samples = scan_images(root, "batch" ,("train", "val", "test"), horizons, prefer_days)
    # print(all_samples)
    by_split: Dict[str, List[ImgSample]] = {"train": [], "val": [], "test": []}
    for s in all_samples: by_split[s.split].append(s)

    # quick summary
    def _count(split): 
        c0 = sum(1 for s in by_split[split] if s.label == 0)
        c1 = sum(1 for s in by_split[split] if s.label == 1)
        return c0, c1, len(by_split[split])

    print(f"[DS] root={root}")
    for split in ("train","val","test"):
        if by_split[split]:
            c0,c1,n = _count(split)
            hs = sorted({s.horizon for s in by_split[split]})
            print(f"[DS] {split:<5} n={n} (y0={c0}, y1={c1}) horizons={hs}")
        else:
            print(f"[DS] {split:<5} n=0")

    # build transforms
    train_tfms, eval_tfms = build_transforms(img_size, normalize=normalize, train_augment=augment)

    # datasets
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        samples = by_split[split]
        if not samples:
            continue
        ds = ImageOnlyDataset(samples, transform=(train_tfms if split=="train" else eval_tfms),
                              return_meta=return_meta, force_rgb=True)

        if split == "train" and balanced:
            w = compute_class_weights(samples)
            per_sample_w = [w[s.label].item() for s in samples]
            sampler = WeightedRandomSampler(per_sample_w, num_samples=len(per_sample_w), replacement=True)
            ld = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=nw,
                            pin_memory=True, drop_last=False)
        else:
            ld = DataLoader(ds, batch_size=bs, shuffle=(split=="train"), num_workers=nw,
                            pin_memory=True, drop_last=False)
        loaders[split] = ld

    if not loaders:
        raise RuntimeError("No loaders built. Check that your dataset exists on disk.")
    return loaders


# ---------------------------
# CLI entrypoint (optional)
# ---------------------------

def _parse_cli():
    p = argparse.ArgumentParser(description="Build PyTorch DataLoaders from YAML (images-only).")
    p.add_argument("--config", type=str, default="config/dataset.yaml",
                   help="Path to YAML config.")
    p.add_argument("--inspect", action="store_true",
                   help="Print a sample batch shape for each available split.")
    return p.parse_args()

def main():
    args = _parse_cli()
    loaders = make_dataloaders_from_yaml(args.config)

    if args.inspect:
        for split, ld in loaders.items():
            batch = next(iter(ld))
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, meta = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                xb, yb = batch; meta = None
            else:
                xb, yb, meta = batch, None, None
            print(f"[BATCH] {split:<5} x={tuple(xb.shape)} y={tuple(yb.shape)}")
            if meta is not None and isinstance(meta, dict):
                # show first 2 paths
                paths = meta.get("path", [])[:2]
                print(f"[META] {split:<5} paths(sample)={paths}")
    else:
        print("[OK] DataLoaders are ready. Use this module from your training script.")

if __name__ == "__main__":
    main()