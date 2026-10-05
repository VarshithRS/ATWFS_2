"""IDD-style mIoU evaluation, following the AutoNUE ``evaluation/evaluate_mIoU.py`` method
(github.com/AutoNUE/public-code, read directly from the repo).

Official method, reproduced here:
  * a (K+1) x (K+1) confusion matrix over level-3 classes; label 255 (or id >= K) in GT *or* prediction goes into an
    extra "ignore / misc" bucket (index K);
  * IoU_c = TP_c / (TP_c + FP_c + FN_c) with FN_c = row sum - TP and FP_c = column sum over ALL other rows,
    INCLUDING the ignore row (GT-ignored pixels predicted as class c count as false positives);
  * mIoU = mean of the K real class IoUs (the bucket is not scored).
Deviation (documented): a class that never occurs in GT or prediction gets NaN and is skipped (``nanmean``)
instead of the official script's divide-by-zero.

This project's model is BINARY (drivable / non-drivable), so :func:`evaluate_binary_idd` applies the same protocol
with K = 2 using the level-3 -> binary collapse. :func:`evaluate_png_folders` applies the unmodified 26-class
protocol to folders of GT/prediction PNGs (useful if you later add a full level-3 model).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from data.dataset import build_dataloader, build_dataset
from data.labels import IDD_LEVEL3_NAMES, NUM_LEVEL3_CLASSES
from models.unet_mobilenet import evidential_probs
from training.metrics import ECEAccumulator
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path, smoke_config
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)


def official_confusion(gt: np.ndarray, pred: np.ndarray, num_classes: int, mat: Optional[np.ndarray] = None) -> np.ndarray:
    """Accumulate the official (K+1)x(K+1) confusion matrix (vectorised).

    Args:
        gt: GT label array (any shape).
        pred: Prediction array (same shape).
        num_classes: K (26 for IDD level-3, 2 for the binary task).
        mat: Optional matrix to accumulate into.

    Returns:
        ``(K+1, K+1)`` int64 matrix; rows = GT, cols = prediction; index K = ignore bucket.
    """
    k = num_classes
    g = np.where((gt == 255) | (gt >= k), k, gt).astype(np.int64).ravel()
    p = np.where((pred == 255) | (pred >= k), k, pred).astype(np.int64).ravel()
    add = np.bincount(g * (k + 1) + p, minlength=(k + 1) ** 2).reshape(k + 1, k + 1)
    return add if mat is None else mat + add


def official_ious(mat: np.ndarray) -> np.ndarray:
    """Per-class IoU with the official FP/FN definition (ignore row counts toward FP).

    Args:
        mat: ``(K+1, K+1)`` matrix from :func:`official_confusion`.

    Returns:
        ``(K,)`` IoUs (NaN for classes absent from GT and prediction).
    """
    k = mat.shape[0] - 1
    ious = np.full(k, np.nan)
    for c in range(k):
        tp = int(mat[c, c])
        fn = int(mat[c, :].sum()) - tp
        fp = int(mat[np.arange(k + 1) != c, c].sum())
        denom = tp + fp + fn
        if denom > 0:
            ious[c] = tp / denom
    return ious


@torch.no_grad()
def evaluate_binary_idd(model: torch.nn.Module, loader: Any, device: torch.device, ece_bins: int = 15) -> Dict[str, float]:
    """Official-protocol mIoU for the binary model (+ terrain accuracy and ECE).

    Args:
        model: Model returning the standard output dict.
        loader: Loader whose batches contain ``mask``, ``valid`` (and ``terrain``).
        device: Device.
        ece_bins: ECE bins.

    Returns:
        Dict with ``miou_idd``, ``iou_nondrivable``, ``iou_drivable``, ``terrain_acc``, ``ece``.
    """
    model.eval()
    mat = np.zeros((3, 3), np.int64)
    ece, t_ok, t_n = ECEAccumulator(ece_bins), 0, 0
    for b in loader:
        out = model(b["image"].to(device))
        pred = out["seg_logits"].argmax(1).cpu().numpy()
        gt = np.where(b["valid"].numpy() > 0, b["mask"].numpy(), 255)  # ignore-label pixels -> bucket
        mat = official_confusion(gt, pred, 2, mat)
        ece.update(evidential_probs(out["alpha"].float()).cpu(), b["mask"])
        v = b["terrain"] >= 0
        t_ok += int((out["terrain_logits"].argmax(1).cpu()[v] == b["terrain"][v]).sum()); t_n += int(v.sum())
    ious = official_ious(mat)
    return {"miou_idd": float(np.nanmean(ious)), "iou_nondrivable": float(ious[0]), "iou_drivable": float(ious[1]),
            "terrain_acc": float(t_ok / t_n) if t_n else float("nan"), "ece": ece.compute()}


def evaluate_png_folders(gt_dir: str, pred_dir: str, suffix: str = "_gtFine_labellevel3Ids.png") -> Dict[str, Any]:
    """Unmodified 26-class AutoNUE protocol on folders of PNGs (``{drive}/{id}{suffix}`` in both trees).

    Args:
        gt_dir: GT root (e.g. ``gtFine/val``).
        pred_dir: Prediction root with the same layout and file names.
        suffix: File suffix.

    Returns:
        ``{"miou": float, "per_class": {name: iou}}``.
    """
    mat = np.zeros((NUM_LEVEL3_CLASSES + 1,) * 2, np.int64)
    files = sorted(Path(gt_dir).glob(f"*/*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No '*{suffix}' files under {gt_dir}")
    for f in files:
        pf = Path(pred_dir) / f.parent.name / f.name
        g = np.array(Image.open(f)); p = np.array(Image.open(pf).resize(g.shape[::-1], Image.NEAREST))
        mat = official_confusion(g, p, NUM_LEVEL3_CLASSES, mat)
    ious = official_ious(mat)
    return {"miou": float(np.nanmean(ious)), "per_class": {n: float(v) for n, v in zip(IDD_LEVEL3_NAMES, ious)}}


def run_evaluation(cfg: Dict[str, Any], checkpoint: Optional[str], synthetic: bool = False, split: Optional[str] = None,
                   out_dir: Optional[str] = None, model: Optional[torch.nn.Module] = None) -> Dict[str, float]:
    """Evaluate a checkpoint on a split and write ``evaluation.json``/``evaluation.csv``.

    Args:
        cfg: Full config.
        checkpoint: Checkpoint path (ignored if ``model`` is given).
        synthetic: Use synthetic data.
        split: Split name (default ``data.eval_split``; ``val`` for synthetic). NOTE: public IDD test has no GT.
        out_dir: Output folder (default ``paths.report_dir``).
        model: Optional ready model.

    Returns:
        Metrics dict.
    """
    set_global_seed(cfg["seed"])
    device = get_device()
    if model is None:
        model, _ = load_model_from_checkpoint(str(checkpoint), device)
    split = "val" if synthetic else (split or cfg["data"]["eval_split"])
    loader = build_dataloader(build_dataset(cfg, split, synthetic), cfg, False)
    m = evaluate_binary_idd(model.to(device), loader, device, cfg["train"]["ece_bins"])
    out = Path(out_dir) if out_dir else resolve_path(cfg["paths"]["report_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(m, indent=2))
    with open(out / "evaluation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["metric", "value"]); w.writerows(m.items())
    logger.info("Evaluation (%s): %s", split, {k: round(v, 4) for k, v in m.items()})
    return m


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="IDD-protocol evaluation.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    setup_logging()
    run_evaluation(load_config(a.config), a.checkpoint, a.synthetic, a.split)


if __name__ == "__main__":
    main()
