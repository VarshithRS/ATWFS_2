"""Failure taxonomy logger: automatically tags every wrong prediction with heuristic failure categories.

Tags (a frame can carry several):
  terrain_misclassification : terrain head disagrees with a (known) terrain label
  missed_obstacle           : > ``missed_obstacle_frac_thr`` of GT-obstacle pixels (person/vehicle ids) are predicted drivable
  low_mIoU_frame            : the frame's own 2-class mIoU < ``low_miou_thr``
  boundary_error            : frame IoU < ``good_iou_thr`` and >= ``boundary_frac_thr`` of the wrong pixels lie within
                              ``boundary_band_px`` of the GT drivable boundary (edges slightly off, region roughly right)
A frame with no tag is "ok". Outputs: per-frame CSV, summary CSV, and (optionally) the failing frames as PNGs for Grad-CAM.
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from data.dataset import build_dataloader, build_dataset
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path, smoke_config
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)
TAGS = ["terrain_misclassification", "missed_obstacle", "low_mIoU_frame", "boundary_error"]


def tag_sample(pred: np.ndarray, gt: np.ndarray, obstacle: np.ndarray, valid: np.ndarray, terrain_pred: int,
               terrain_gt: int, ev: Dict[str, Any]) -> Dict[str, Any]:
    """Tag one frame.

    Args:
        pred: ``(H, W)`` predicted binary mask.
        gt: ``(H, W)`` GT binary mask.
        obstacle: ``(H, W)`` GT obstacle mask.
        valid: ``(H, W)`` 1 where GT is labeled.
        terrain_pred: Predicted terrain class.
        terrain_gt: GT terrain class (-1 = unknown).
        ev: ``evaluation`` config section.

    Returns:
        Dict with ``tags`` (list), ``iou_drivable``, ``frame_miou``, ``boundary_frac``, ``obstacle_miss_frac``.
    """
    v = valid > 0
    p, g = pred[v] > 0, gt[v] > 0
    ious = []
    for cls in (True, False):
        u = ((p == cls) | (g == cls)).sum()
        if u > 0:
            ious.append(((p == cls) & (g == cls)).sum() / u)
    frame_miou = float(np.mean(ious)) if ious else 1.0
    u = (p | g).sum()
    iou_d = float((p & g).sum() / u) if u > 0 else 1.0
    wrong = (pred != gt) & v
    n_wrong = int(wrong.sum())
    edge = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    k = 2 * int(ev["boundary_band_px"]) + 1
    band = cv2.dilate(edge, np.ones((k, k), np.uint8)) > 0
    bfrac = float((wrong & band).sum() / n_wrong) if n_wrong else 0.0
    ob = obstacle > 0
    miss = float(((pred > 0) & ob).sum() / ob.sum()) if ob.sum() >= ev["min_obstacle_px"] else 0.0
    tags: List[str] = []
    if terrain_gt >= 0 and terrain_pred != terrain_gt:
        tags.append("terrain_misclassification")
    if miss > ev["missed_obstacle_frac_thr"]:
        tags.append("missed_obstacle")
    if frame_miou < ev["low_miou_thr"]:
        tags.append("low_mIoU_frame")
    if n_wrong > 0 and frame_miou < ev["good_iou_thr"] and bfrac >= ev["boundary_frac_thr"]:
        tags.append("boundary_error")
    return {"tags": tags, "iou_drivable": iou_d, "frame_miou": frame_miou, "boundary_frac": bfrac, "obstacle_miss_frac": miss}


@torch.no_grad()
def run_taxonomy(cfg: Dict[str, Any], model: torch.nn.Module, loader: Any, device: torch.device,
                 out_dir: Optional[str] = None, save_failures: bool = False) -> Dict[str, Any]:
    """Tag every frame of a loader and write ``failure_frames.csv`` + ``failure_summary.csv``.

    Args:
        cfg: Full config.
        model: Model.
        loader: Loader (batches with ``mask``, ``obstacle``, ``valid``, ``terrain``, ``image``, ``index``).
        device: Device.
        out_dir: Output directory (default ``paths.report_dir``).
        save_failures: Also dump failing frames (RGB PNG) to ``<out_dir>/failures/`` for Grad-CAM.

    Returns:
        ``{"rows": [...], "summary": {tag: count}, "frames_csv": str, "summary_csv": str}``.
    """
    ev = cfg["evaluation"]
    out = Path(out_dir) if out_dir else resolve_path(cfg["paths"]["report_dir"])
    out.mkdir(parents=True, exist_ok=True)
    mean, std = np.asarray(cfg["data"]["imagenet_mean"]), np.asarray(cfg["data"]["imagenet_std"])
    model.eval()
    rows, saved = [], 0
    for b in loader:
        o = model(b["image"].to(device))
        preds = o["seg_logits"].argmax(1).cpu().numpy()
        tp = o["terrain_logits"].argmax(1).cpu().numpy()
        for i in range(preds.shape[0]):
            r = tag_sample(preds[i], b["mask"][i].numpy(), b["obstacle"][i].numpy(), b["valid"][i].numpy(),
                           int(tp[i]), int(b["terrain"][i]), ev)
            idx = int(b["index"][i])
            rows.append({"index": idx, "status": "wrong" if r["tags"] else "ok", "tags": "|".join(r["tags"]),
                         **{k: round(r[k], 4) for k in ("iou_drivable", "frame_miou", "boundary_frac", "obstacle_miss_frac")},
                         "terrain_pred": int(tp[i]), "terrain_gt": int(b["terrain"][i])})
            if save_failures and r["tags"] and saved < ev["max_saved_failures"]:
                img = ((b["image"][i].permute(1, 2, 0).numpy() * std + mean).clip(0, 1) * 255).astype(np.uint8)
                (out / "failures").mkdir(exist_ok=True)
                cv2.imwrite(str(out / "failures" / f"frame_{idx:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                saved += 1
    counts = Counter(t for r in rows for t in r["tags"].split("|") if t)
    summary = {t: counts.get(t, 0) for t in TAGS}
    n_wrong = sum(r["status"] == "wrong" for r in rows)
    frames_csv, summary_csv = out / "failure_frames.csv", out / "failure_summary.csv"
    with open(frames_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["tag", "count", "fraction_of_frames"])
        for t in TAGS:
            w.writerow([t, summary[t], round(summary[t] / max(1, len(rows)), 4)])
        w.writerow(["any_failure", n_wrong, round(n_wrong / max(1, len(rows)), 4)])
        w.writerow(["total_frames", len(rows), 1.0])
    logger.info("Failure taxonomy over %d frames: %s (wrong frames: %d)", len(rows), summary, n_wrong)
    return {"rows": rows, "summary": summary, "frames_csv": str(frames_csv), "summary_csv": str(summary_csv)}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Failure taxonomy over a split.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--split", default=None)
    ap.add_argument("--save-failures", action="store_true")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    set_global_seed(cfg["seed"])
    dev = get_device()
    model, _ = load_model_from_checkpoint(a.checkpoint, dev)
    split = "val" if a.synthetic else (a.split or cfg["data"]["eval_split"])
    run_taxonomy(cfg, model, build_dataloader(build_dataset(cfg, split, a.synthetic), cfg, False), dev, save_failures=a.save_failures)


if __name__ == "__main__":
    main()
