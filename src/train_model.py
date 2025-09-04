# src/train_from_yaml.py
# Train a CNN on the generated image dataset using a YAML config.
# - Loads DataLoaders via src/torch_loader.py (images-only path)
# - Model: ResNet18/ResNet50 (configurable)
# - Optimizer: AdamW/SGD (configurable), LR scheduler optional
# - AMP mixed-precision, early stopping, checkpointing
# - Metrics: accuracy, precision, recall, F1 (binary)

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import time
import json
import random
import numpy as np
from contextlib import nullcontext
# import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as tvm

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

from .torch_loader import make_dataloaders_from_yaml  

# ---------------------------
# Config helpers
# ---------------------------

def _get(d: dict, path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# ---------------------------
# Reproducibility & device
# ---------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster on GPUs with varying shapes

def pick_device(pref: str = "auto") -> torch.device:
    if pref and pref.lower() != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

def make_amp_ctx_and_scaler(device, use_amp: bool = True, allow_cpu_amp: bool = False):
    """
    Return (autocast_context, scaler) compatible with many PyTorch versions.
    - CUDA: autocast + GradScaler
    - CPU (optionnel): autocast si allow_cpu_amp=True
    - MPS/others: pas d'autocast, pas de scaler (nullcontext)
    """
    devt = device.type  # 'cuda' | 'mps' | 'cpu'

    # Pas d'AMP demandé
    if not use_amp:
        return nullcontext(), None

    # ----- CUDA -----
    if devt == "cuda":
        # autocast
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                ctx = torch.amp.autocast(device_type="cuda", enabled=True)
            except TypeError:
                ctx = torch.cuda.amp.autocast(enabled=True)
        else:
            ctx = torch.cuda.amp.autocast(enabled=True)
        # scaler
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                scaler = torch.amp.GradScaler()   # pas d'arg 'device_type' -> compat large
            except TypeError:
                scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = torch.cuda.amp.GradScaler()
        return ctx, scaler

    # ----- CPU (optionnel) -----
    if devt == "cpu" and allow_cpu_amp:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                ctx = torch.amp.autocast(device_type="cpu", enabled=True)
            except TypeError:
                ctx = nullcontext()
        else:
            ctx = nullcontext()
        return ctx, None

    # ----- MPS ou autre : PAS d'autocast -----
    return nullcontext(), None
# ---------------------------
# Model builder
# ---------------------------

def build_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = (name or "resnet18").lower()
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
    if name == "inception_v3":
        m = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None, aux_logits=False)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m
    raise ValueError(f"Unknown model '{name}'. Supported: resnet18, resnet50")

# ---------------------------
# Metrics (binary)
# ---------------------------

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    # logits: [B,2], targets: [B] in {0,1}
    preds = logits.argmax(dim=1)
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}

def update_confusion_counts(logits: torch.Tensor, targets: torch.Tensor, acc):
    """Accumulate confusion counts in-place in dict acc={'tp':..,'tn':..,'fp':..,'fn':..}."""
    preds = logits.argmax(dim=1)
    acc["tp"] += int(((preds == 1) & (targets == 1)).sum().item())
    acc["tn"] += int(((preds == 0) & (targets == 0)).sum().item())
    acc["fp"] += int(((preds == 1) & (targets == 0)).sum().item())
    acc["fn"] += int(((preds == 0) & (targets == 1)).sum().item())

def format_confusion(cm: dict, normalize: bool = False) -> str:
    """
    Return a pretty 2x2 matrix as string:
      [[TN, FP],
       [FN, TP]]
    If normalize=True, each cell is divided by total.
    """
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    if normalize:
        tot = max(1, tn + fp + fn + tp)
        tn, fp, fn, tp = (x / tot for x in (tn, fp, fn, tp))
        return (f"[[{tn:.3f}, {fp:.3f}],\n"
                f" [{fn:.3f}, {tp:.3f}]]")
    return (f"[[{tn}, {fp}],\n [{fn}, {tp}]]")


# ---------------------------
# Train / evaluate loops
# ---------------------------

def run_epoch(model, loader, criterion, optimizer=None,
              device=torch.device("cpu"), use_amp=True):
    """
    One epoch over `loader`.
    - Train if `optimizer` is not None, else eval.
    - Uses `make_amp_ctx_and_scaler(device, use_amp)` for AMP compat
      across CUDA / MPS / CPU and PyTorch versions.
    - Returns: (avg_loss, metrics_dict, confusion_dict)
    """
    is_train = optimizer is not None
    model.train(is_train)

    # metrics accumulators
    total_loss = 0.0
    m_sum = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_batches = 0
    cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    # Create a single scaler for the whole epoch (if CUDA+AMP).
    # We'll create a fresh autocast context *per batch*.
    _, scaler = make_amp_ctx_and_scaler(device, use_amp)

    for batch in loader:
        # support (x, y) or (x, y, meta)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            xb, yb = batch[0], batch[1]
        else:
            xb, yb = batch

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # fresh autocast context each iteration (compat new/old PyTorch)
        ctx, _ = make_amp_ctx_and_scaler(device, use_amp)
        with ctx:
            logits = model(xb)
            loss = criterion(logits, yb)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        update_confusion_counts(logits.detach(), yb, cm)
        m = compute_metrics(logits.detach(), yb)
        for k in m_sum:
            m_sum[k] += m[k]
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    metrics = {k: (m_sum[k] / max(1, n_batches)) for k in m_sum}
    return avg_loss, metrics, cm

# ---------------------------
# Checkpointing / early stopping
# ---------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_monitor, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_json = json.loads(json.dumps(cfg, default=str))
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_monitor": best_monitor,
        "config": cfg_json,  
    }, path)

# ---------------------------
# Main training entry
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train a CNN from YAML config.")
    ap.add_argument("--config", type=str, default="config/dataset.yaml", help="Path to YAML config.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

    # Repro & device
    seed = int(_get(cfg, "training.seed", 42)); set_seed(seed)
    device = pick_device(_get(cfg, "training.device", "auto"))

    # Data
    loaders = make_dataloaders_from_yaml(cfg_path)
    train_loader = loaders.get("train")
    val_loader   = loaders.get("val")
    test_loader  = loaders.get("test")

    # Model
    model_name = _get(cfg, "training.model", "resnet18")
    pretrained = bool(_get(cfg, "training.pretrained", True))
    num_classes = int(_get(cfg, "training.num_classes", 2))
    model = build_model(model_name, num_classes=num_classes, pretrained=pretrained).to(device)

    # Loss (class weights optional)
    use_class_weights = bool(_get(cfg, "training.class_weights", False))
    ce_weight = None
    if use_class_weights and train_loader is not None:
        # compute inverse-freq weights from the dataset already loaded
        # the dataset stores labels in dataset.samples (ImageOnlyDataset)
        counts = {0: 0, 1: 0}
        for s in getattr(train_loader.dataset, "samples", []):
            counts[int(s.label)] += 1
        total = counts[0] + counts[1]
        w0 = total / (2.0 * max(1, counts[0])); w1 = total / (2.0 * max(1, counts[1]))
        ce_weight = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=ce_weight)

    # Optimizer
    opt_name = (_get(cfg, "training.optimizer", "adamw") or "adamw").lower()
    lr = float(_get(cfg, "training.lr", 1e-3))
    wd = float(_get(cfg, "training.weight_decay", 1e-4))
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    else:
        raise ValueError("training.optimizer must be 'adamw' or 'sgd'.")

    # Scheduler
    sch_name = (_get(cfg, "training.scheduler", "none") or "none").lower()
    scheduler = None
    max_epochs = int(_get(cfg, "training.epochs", 20))
    if sch_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif sch_name == "step":
        step_size = int(_get(cfg, "training.step_size", 10))
        gamma = float(_get(cfg, "training.gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sch_name == "none":
        scheduler = None
    else:
        raise ValueError("training.scheduler must be 'none' | 'cosine' | 'step'.")

    # Misc training params
    use_amp = bool(_get(cfg, "training.use_amp", True))
    monitor = (_get(cfg, "training.monitor", "val_f1") or "val_f1").lower()  # 'val_loss' or 'val_f1'
    patience = int(_get(cfg, "training.patience", 7))
    save_dir = Path(_get(cfg, "training.save_dir", "runs/exp"))
    show_cm = bool(_get(cfg, "training.show_confusion", True))
    norm_cm = bool(_get(cfg, "training.normalize_confusion", False))
    save_best = save_dir / "best.pt"
    save_last = save_dir / "last.pt"

    print(f"[CFG] model={model_name} pretrained={pretrained} device={device} "
          f"epochs={max_epochs} lr={lr} opt={opt_name} sch={sch_name} use_amp={use_amp}")
    print(f"[CFG] monitor={monitor} patience={patience} class_weights={use_class_weights} save_dir={save_dir}")

    # ---------------- Train loop ----------------
    best_monitor = float("inf") if monitor == "val_loss" else -float("inf")
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_metrics, tr_cm = run_epoch(model, train_loader, criterion, optimizer,
                                              device=device, use_amp=use_amp)
        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            va_loss, va_metrics, va_cm = run_epoch(model, val_loader, criterion, optimizer=None,
                                       device=device, use_amp=use_amp) if val_loader else (float("nan"), {"acc": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}, {"tp":0,"tn":0,"fp":0,"fn":0})

        else:
            va_loss, va_metrics = float("nan"), {"acc": float("nan"), "precision": float("nan"),
                                                 "recall": float("nan"), "f1": float("nan")}
        dt = time.time() - t0

        print(f"[E{epoch:03d}] "
            f"train: loss={tr_loss:.4f} acc={tr_metrics['acc']:.3f} f1={tr_metrics['f1']:.3f} | "
            f"val: loss={va_loss:.4f} acc={va_metrics['acc']:.3f} f1={va_metrics['f1']:.3f} "
            f"({dt:.1f}s)")
        
        if show_cm:
            print("[CM][train]\n" + format_confusion(tr_cm, normalize=norm_cm))
            if val_loader is not None:
                print("[CM][val]\n" + format_confusion(va_cm, normalize=norm_cm))
        # checkpoint last
        save_checkpoint(save_last, model, optimizer, scheduler, epoch, best_monitor, cfg)

        # early stopping on monitor
        cur = (-va_loss) if monitor == "val_loss" else va_metrics["f1"]
        improved = (cur > best_monitor)
        if improved:
            best_monitor = cur
            bad_epochs = 0
            save_checkpoint(save_best, model, optimizer, scheduler, epoch, best_monitor, cfg)
            print(f"  ↳ [best] saved to {save_best}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[EARLY] No improvement for {patience} epochs — stopping.")
                break

    # ---------------- Test (optional) ----------------
    if test_loader is not None and save_best.exists():
        ckpt = torch.load(save_best, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        te_loss, te_metrics, te_cm = run_epoch(model, test_loader, criterion, optimizer=None,
                                            device=device, use_amp=use_amp)
        print(f"[TEST] loss={te_loss:.4f} acc={te_metrics['acc']:.3f} "
            f"prec={te_metrics['precision']:.3f} rec={te_metrics['recall']:.3f} "
            f"f1={te_metrics['f1']:.3f}")
        if show_cm:
            print("[CM][test]\n" + format_confusion(te_cm, normalize=norm_cm))

if __name__ == "__main__":
    main()