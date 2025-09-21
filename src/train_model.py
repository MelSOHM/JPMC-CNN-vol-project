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
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

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

def _resnet_custom(blocks, num_classes=2, dropout=0.5, bottleneck=False):
    """
    blocks = [b1, b2, b3, b4] nombre de BasicBlocks (ou Bottleneck) par stage.
    Ex: resnet18 = [2,2,2,2]. Ici on propose plus petit: [1,1,1,1] etc.
    """
    Block = Bottleneck if bottleneck else BasicBlock
    m = ResNet(Block, blocks, num_classes=num_classes)  # pas de prétrain pour ces profondeurs
    in_feats = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, num_classes))
    return m

def resnet8(num_classes=2, dropout=0.5):   # 2*sum(blocks)=8 couches (BasicBlock)
    return _resnet_custom([1,1,1,1], num_classes=num_classes, dropout=dropout, bottleneck=False)

def resnet10(num_classes=2, dropout=0.5):
    # 10 ~ [1,1,1,2] (approximation simple)
    return _resnet_custom([1,1,1,2], num_classes=num_classes, dropout=dropout, bottleneck=False)

def resnet14(num_classes=2, dropout=0.5):
    return _resnet_custom([1,1,2,3], num_classes=num_classes, dropout=dropout, bottleneck=False)

def build_model(name: str, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    name = (name or "resnet18").lower()
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
        return m
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
        return m
    if name == "inception_v3":
        m = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None, aux_logits=False)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
        return m
    
    # --- ResNet courts (no pre-train)
    if name == "resnet8":
        return resnet8(num_classes=num_classes, dropout=dropout)
    if name == "resnet10":
        return resnet10(num_classes=num_classes, dropout=dropout)
    if name == "resnet14":
        return resnet14(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model '{name}'. Supported: resnet18, resnet50")

# ---------------------------
# Metrics (multi-class)
# ---------------------------

@torch.no_grad()
def confusion_matrix_from_logits(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Retourne une matrice de confusion [K,K] avec K=num_classes.
    L'axe 0 = vrai (rows), axe 1 = préd (cols).
    """
    preds = logits.argmax(dim=1)
    t = targets.view(-1).to(torch.int64)
    p = preds.view(-1).to(torch.int64)

    # filtrer les labels hors [0..K-1] par sécurité
    mask = (t >= 0) & (t < num_classes)
    t = t[mask]; p = p[mask]

    cm = torch.bincount(t * num_classes + p, minlength=num_classes * num_classes)
    cm = cm.view(num_classes, num_classes)
    return cm

@torch.no_grad()
def compute_multiclass_metrics(cm: torch.Tensor) -> Dict[str, float]:
    """
    À partir d'une matrice de confusion KxK, calcule:
    - accuracy
    - macro_precision, macro_recall, macro_f1
    - micro_precision (=acc), micro_recall, micro_f1 (=acc)
    """
    K = cm.shape[0]
    tp = cm.diag()
    support = cm.sum(dim=1)           # vrais par classe
    pred_pos = cm.sum(dim=0)          # prédits par classe
    total = cm.sum()

    acc = (tp.sum() / total).item() if total > 0 else 0.0

    prec_k = tp / torch.clamp(pred_pos, min=1)
    rec_k  = tp / torch.clamp(support,  min=1)
    f1_k   = 2 * prec_k * rec_k / torch.clamp(prec_k + rec_k, min=1e-8)

    macro_precision = prec_k.mean().item()
    macro_recall    = rec_k.mean().item()
    macro_f1        = f1_k.mean().item()

    # micro = global
    micro_precision = acc
    micro_recall    = acc
    micro_f1        = acc

    return {
        "acc": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }

def format_confusion(cm: torch.Tensor, normalize: bool = False) -> str:
    """
    Affiche joliment une matrice KxK.
    Rows = vrais, Cols = prédits.
    """
    if normalize:
        tot = cm.sum().item()
        M = (cm.float() / max(1.0, tot)).cpu().numpy()
        rows = ["[" + ", ".join(f"{x:.3f}" for x in r) + "]" for r in M]
    else:
        M = cm.cpu().numpy().astype(int)
        rows = ["[" + ", ".join(str(int(x)) for x in r) + "]" for r in M]
    return "[\n " + ",\n ".join(rows) + "\n]"

# ---------------------------
# Train / evaluate loops
# ---------------------------
def _normalize_targets(y: torch.Tensor) -> torch.Tensor:
    """
    CrossEntropyLoss exige des labels dans [0..C-1].
    Si on reçoit {-1,0,1}, on décale de +1 -> {0,1,2}.
    Si déjà non-négatifs, on laisse tel quel.
    """
    if y.numel() == 0:
        return y
    if torch.min(y) < 0:
        return y + int(-torch.min(y))
    return y

def run_epoch(model, loader, criterion, optimizer=None,
              device=torch.device("cpu"), use_amp=True):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0

    # on accumule la CM sur tout l'epoch
    num_classes = getattr(model.fc[-1], "out_features", None) if hasattr(model, "fc") else None
    if num_classes is None:
        # fallback robuste
        # on fera un premier batch pour inférer K
        num_classes = -1
    cm_accum = None

    _, scaler = make_amp_ctx_and_scaler(device, use_amp)

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            xb, yb = batch[0], batch[1]
        else:
            xb, yb = batch

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        yb = _normalize_targets(yb)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        ctx, _ = make_amp_ctx_and_scaler(device, use_amp)
        with ctx:
            logits = model(xb)
            if num_classes == -1:
                num_classes = logits.shape[1]
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
        cm_batch = confusion_matrix_from_logits(logits.detach(), yb, num_classes)
        cm_accum = cm_batch if cm_accum is None else (cm_accum + cm_batch)
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    metrics = compute_multiclass_metrics(cm_accum if cm_accum is not None else torch.zeros((num_classes, num_classes)))
    return avg_loss, metrics, cm_accum


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
    dropout = float(_get(cfg, "training.dropout", 0.5))
    model = build_model(model_name, num_classes=num_classes, pretrained=pretrained, dropout=dropout).to(device)

    # Loss (class weights optional)
    use_class_weights = bool(_get(cfg, "training.class_weights", False))
    ce_weight = None
    if use_class_weights and train_loader is not None:
        # Essaye d'inférer K depuis le modèle ou dataset
        num_classes = int(_get(cfg, "training.num_classes", 3))
        counts = torch.zeros(num_classes, dtype=torch.long)
        for s in getattr(train_loader.dataset, "samples", []):
            y = int(s.label)
            if y < 0:
                y += 1  # map {-1,0,1} -> {0,1,2}
            if 0 <= y < num_classes:
                counts[y] += 1
        total = int(counts.sum().item())
        # inverse-frequency (stabilisé)
        ce_weight = (total / (num_classes * torch.clamp(counts, min=1))).to(torch.float32).to(device)
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
        f"train: loss={tr_loss:.4f} acc={tr_metrics['acc']:.3f} macroF1={tr_metrics['macro_f1']:.3f} | "
        f"val: loss={va_loss:.4f} acc={va_metrics['acc']:.3f} macroF1={va_metrics['macro_f1']:.3f} "
        f"({dt:.1f}s)")
        
        if show_cm:
            print("[CM][train]\n" + format_confusion(tr_cm, normalize=norm_cm))
            if val_loader is not None:
                print("[CM][val]\n" + format_confusion(va_cm, normalize=norm_cm))
        # checkpoint last
        save_checkpoint(save_last, model, optimizer, scheduler, epoch, best_monitor, cfg)

        # early stopping on monitor
        cur = (-va_loss) if monitor == "val_loss" else va_metrics.get("macro_f1", va_metrics.get("f1", float("nan")))
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
            f"macroP={te_metrics['macro_precision']:.3f} macroR={te_metrics['macro_recall']:.3f} "
            f"macroF1={te_metrics['macro_f1']:.3f}")
        if show_cm:
            print("[CM][test]\n" + format_confusion(te_cm, normalize=norm_cm))

if __name__ == "__main__":
    main()