"""Training script (teacher model) + reusable training loop.

Usage
-----
    python -m training.train --config config.yaml                 # real IDD (GPU recommended)
    python -m training.train --cpu-fallback                       # 2-epoch synthetic sanity run (CPU, <1 min)
    python -m training.train --synthetic --epochs 3               # synthetic data, config sizes

# %% [markdown]  ----- COLAB / KAGGLE CELL BLOCK (copy into the first notebook cells) -----
# %% cell 1
#   !pip -q install -r requirements.txt
# %% cell 2
#   from training.train import colab_setup; cfg_overrides = colab_setup()
# %% cell 3
#   !python -m training.train --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import build_dataloader, build_dataset
from models.unet_mobilenet import MultiTaskUNet, build_model, evidential_probs
from training.losses import MultiTaskLoss, batch_boundary_weights
from training.metrics import ECEAccumulator, confusion_update, summarize
from utils.config import load_config, resolve_path, smoke_config
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)

ExtraLoss = Callable[[Dict[str, torch.Tensor], Dict[str, Any]], Tuple[torch.Tensor, Dict[str, float]]]


# ------------------------------------------------------------------ Colab helper
def colab_setup(drive_data_path_env: str = "IDD_DRIVE_PATH") -> Dict[str, Any]:
    """Colab/Kaggle setup: GPU check + (Colab only) Google-Drive mount.

    IDD requires registration, so it cannot be auto-downloaded: put the dataset in your Drive and
    set the env var ``IDD_DRIVE_PATH`` (e.g. ``/content/drive/MyDrive/idd``), or edit
    ``paths.data_root`` in config.yaml.

    Args:
        drive_data_path_env: Name of the env var holding the dataset path.

    Returns:
        Config overrides dict (``paths.data_root`` if the env var is set).
    """
    setup_logging()
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("No GPU detected. In Colab: Runtime > Change runtime type > GPU.")
    if "google.colab" in sys.modules:  # pragma: no cover - only on Colab
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    p = os.environ.get(drive_data_path_env)
    return {"paths": {"data_root": p}} if p else {}


# ------------------------------------------------------------------ checkpoint IO
def save_checkpoint(path: Path, model: nn.Module, criterion: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    epoch: int, metrics: Dict[str, float], model_cfg: Dict[str, Any]) -> None:
    """Save a checkpoint.

    Args:
        path: Destination file.
        model: Model.
        criterion: Loss module (holds learnable task weights).
        optimizer: Optimizer (optional).
        epoch: Epoch index.
        metrics: Latest validation metrics.
        model_cfg: Model config section needed to rebuild the model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "epoch": epoch, "metrics": metrics, "model_cfg": model_cfg}, path)


def load_model_from_checkpoint(path: str, device: torch.device) -> Tuple[MultiTaskUNet, Dict[str, Any]]:
    """Rebuild a model from a checkpoint and load its weights.

    Args:
        path: Checkpoint path.
        device: Target device.

    Returns:
        ``(model in eval mode, checkpoint dict)``.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = dict(ck["model_cfg"]); mc["pretrained"] = False
    model = build_model(mc).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: Optional[nn.Module] = None,
             ece_bins: int = 15) -> Dict[str, float]:
    """Evaluate segmentation mIoU, terrain accuracy and uncertainty calibration error.

    Args:
        model: Model (any model returning the standard output dict).
        loader: Validation loader.
        device: Device.
        criterion: Optional loss to also report ``val_loss``.
        ece_bins: Bins for ECE.

    Returns:
        Metrics dict.
    """
    model.eval()
    k = 2
    conf = torch.zeros(k, k, dtype=torch.long)
    ece = ECEAccumulator(ece_bins)
    t_ok = t_n = 0
    loss_sum, n_batches = 0.0, 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["mask"].to(device)
        out = model(x)
        pred = out["seg_logits"].argmax(1)
        conf = confusion_update(conf, pred.cpu(), y.cpu(), k)
        ece.update(evidential_probs(out["alpha"].float()), y)
        t = batch["terrain"].to(device)
        valid = t >= 0
        t_ok += int((out["terrain_logits"].argmax(1)[valid] == t[valid]).sum())
        t_n += int(valid.sum())
        if criterion is not None:
            b = {kk: v.to(device) for kk, v in batch.items()}
            loss_sum += float(criterion(out, b)[0]); n_batches += 1
    m = summarize(conf, t_ok, t_n, ece.compute())
    if criterion is not None:
        m["val_loss"] = loss_sum / max(1, n_batches)
    return m


# ------------------------------------------------------------------ main loop
def run_training(cfg: Dict[str, Any], model: nn.Module, criterion: MultiTaskLoss, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device, tag: str, model_cfg: Dict[str, Any],
                 epochs: Optional[int] = None, extra_loss_fn: Optional[ExtraLoss] = None) -> Dict[str, Any]:
    """Generic train/val loop used by teacher training, distillation and prune fine-tuning.

    Args:
        cfg: Full config.
        model: Model to train.
        criterion: Multi-task loss (its learnable weights are optimised too).
        train_loader: Training loader.
        val_loader: Validation loader.
        device: Device.
        tag: Name prefix for checkpoints/curves.
        model_cfg: Model config (stored in the checkpoint for rebuilding).
        epochs: Override number of epochs.
        extra_loss_fn: Optional ``fn(batch_on_device, model_out) -> (extra_loss, info)`` (distillation).

    Returns:
        ``{"history": [...], "best_path": str, "last_path": str}``.
    """
    t = cfg["train"]
    epochs = epochs or t["epochs"]
    model.to(device); criterion.to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_dir = resolve_path(cfg["paths"]["checkpoint_dir"]); report_dir = resolve_path(cfg["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    history, best, best_path, last_path = [], -math.inf, ckpt_dir / f"{tag}_best.pt", ckpt_dir / f"{tag}_last.pt"
    for ep in range(epochs):
        model.train(); criterion.set_epoch(ep)
        t0, run, n = time.time(), 0.0, 0
        for it, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch["bweight"] = batch_boundary_weights(batch["mask"], cfg["loss"]["boundary"]["w0"], cfg["loss"]["boundary"]["sigma"])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(batch["image"])
            out = {k: ([a.float() for a in v] if isinstance(v, list) else v.float()) for k, v in out.items()}
            loss, info = criterion(out, batch)
            if extra_loss_fn is not None:
                extra, einfo = extra_loss_fn(batch, out)
                loss, info = loss + extra, {**info, **einfo}
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {ep} iter {it}: {info}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, t["grad_clip"])
            scaler.step(opt); scaler.update()
            run += float(loss.detach()); n += 1
            if it % t["log_every"] == 0:
                logger.info("[%s] ep %d it %d loss %.4f", tag, ep, it, float(loss.detach()))
        sched.step()
        val = evaluate(model, val_loader, device, criterion, t["ece_bins"])
        rec = {"epoch": ep, "train_loss": run / max(1, n), "time_s": time.time() - t0, **val}
        history.append(rec)
        logger.info("[%s] epoch %d | train_loss %.4f | val_loss %.4f | mIoU %.4f | terrain_acc %.3f | ECE %.4f",
                    tag, ep, rec["train_loss"], val["val_loss"], val["miou"], val["terrain_acc"], val["ece"])
        save_checkpoint(last_path, model, criterion, opt, ep, val, model_cfg)
        score = val["miou"] if not math.isnan(val["miou"]) else -val["val_loss"]
        if score > best:
            best = score
            save_checkpoint(best_path, model, criterion, opt, ep, val, model_cfg)
    _write_curves(history, report_dir, tag)
    return {"history": history, "best_path": str(best_path), "last_path": str(last_path)}


def _write_curves(history: list, report_dir: Path, tag: str) -> None:
    """Write per-epoch metrics to CSV/JSON and (if matplotlib exists) a loss-curve PNG.

    Args:
        history: List of per-epoch dicts.
        report_dir: Output folder.
        tag: Filename prefix.
    """
    with open(report_dir / f"{tag}_history.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
    (report_dir / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot([h["train_loss"] for h in history], label="train"); ax[0].plot([h["val_loss"] for h in history], label="val")
        ax[0].set_title("loss"); ax[0].legend()
        ax[1].plot([h["miou"] for h in history], label="mIoU"); ax[1].plot([h["ece"] for h in history], label="ECE"); ax[1].legend()
        fig.tight_layout(); fig.savefig(report_dir / f"{tag}_curves.png", dpi=110); plt.close(fig)
    except Exception:
        logger.info("matplotlib not available; CSV/JSON curves written only.")


def train(cfg: Dict[str, Any], synthetic: bool = False, tag: str = "teacher", epochs: Optional[int] = None) -> Dict[str, Any]:
    """Build everything from ``cfg`` and train the teacher model.

    Args:
        cfg: Full config.
        synthetic: Use synthetic data instead of IDD.
        tag: Checkpoint name prefix.
        epochs: Optional epoch override.

    Returns:
        Result dict from :func:`run_training`.
    """
    set_global_seed(cfg["seed"])
    device = get_device()
    logger.info("Device: %s", device)
    tr = build_dataloader(build_dataset(cfg, "train", synthetic), cfg, True)
    va = build_dataloader(build_dataset(cfg, "val", synthetic), cfg, False)
    model = build_model(cfg["model"], cfg["loss"]["evidence_clip"])
    crit = MultiTaskLoss(cfg["loss"])
    return run_training(cfg, model, crit, tr, va, device, tag, cfg["model"], epochs)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train the multi-task drivable-region model.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="use procedurally generated data")
    ap.add_argument("--cpu-fallback", action="store_true", help="2-epoch synthetic sanity run (tiny sizes)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default="teacher")
    a = ap.parse_args()
    setup_logging()
    if a.cpu_fallback:
        cfg = smoke_config(); a.synthetic = True; a.tag = "smoke"; a.epochs = cfg["smoke"]["epochs"]
    else:
        cfg = load_config(a.config)
    res = train(cfg, a.synthetic, a.tag, a.epochs)
    logger.info("Done. Best checkpoint: %s", res["best_path"])


if __name__ == "__main__":
    main()
