"""Segmentation / terrain / calibration metrics (accumulator style)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def confusion_update(conf: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, k: int) -> torch.Tensor:
    """Accumulate a ``(k,k)`` confusion matrix (rows = GT, cols = prediction).

    Args:
        conf: Running ``(k,k)`` long matrix.
        pred: Predicted labels (any shape).
        gt: GT labels (same shape).
        k: Number of classes.

    Returns:
        Updated matrix.
    """
    idx = gt.reshape(-1).long() * k + pred.reshape(-1).long()
    return conf + torch.bincount(idx, minlength=k * k).reshape(k, k).to(conf.device)


def ious_from_confusion(conf: torch.Tensor) -> np.ndarray:
    """Per-class IoU ``TP / (TP + FP + FN)``; NaN where the class never occurs.

    Args:
        conf: ``(k,k)`` confusion matrix.

    Returns:
        ``(k,)`` float array.
    """
    c = conf.double().cpu().numpy()
    tp = np.diag(c)
    denom = c.sum(0) + c.sum(1) - tp
    return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)


class ECEAccumulator:
    """Expected Calibration Error over pixel confidences (uncertainty-calibration metric)."""

    def __init__(self, bins: int = 15, max_pixels_per_batch: int = 50_000, seed: int = 0) -> None:
        """Create the accumulator.

        Args:
            bins: Number of confidence bins.
            max_pixels_per_batch: Random subsample size per update (speed).
            seed: Subsampling seed.
        """
        self.bins = bins
        self.max_px = max_pixels_per_batch
        self.gen = torch.Generator().manual_seed(seed)
        self.count = torch.zeros(bins, dtype=torch.double)
        self.conf_sum = torch.zeros(bins, dtype=torch.double)
        self.acc_sum = torch.zeros(bins, dtype=torch.double)

    def update(self, probs: torch.Tensor, gt: torch.Tensor) -> None:
        """Add a batch.

        Args:
            probs: ``(B,K,H,W)`` class probabilities (e.g. Dirichlet mean).
            gt: ``(B,H,W)`` labels.
        """
        conf, pred = probs.max(dim=1)
        conf, correct = conf.reshape(-1).double().cpu(), (pred == gt).reshape(-1).double().cpu()
        if conf.numel() > self.max_px:
            sel = torch.randperm(conf.numel(), generator=self.gen)[: self.max_px]
            conf, correct = conf[sel], correct[sel]
        b = torch.clamp((conf * self.bins).long(), max=self.bins - 1)
        self.count += torch.bincount(b, minlength=self.bins).double()
        self.conf_sum += torch.bincount(b, weights=conf, minlength=self.bins)
        self.acc_sum += torch.bincount(b, weights=correct, minlength=self.bins)

    def compute(self) -> float:
        """Return ECE in [0, 1] (NaN if nothing accumulated).

        Returns:
            ECE value.
        """
        n = self.count.sum().item()
        if n == 0:
            return float("nan")
        m = self.count > 0
        gap = (self.acc_sum[m] / self.count[m] - self.conf_sum[m] / self.count[m]).abs()
        return float((gap * self.count[m]).sum().item() / n)


def summarize(conf: torch.Tensor, terrain_correct: int, terrain_total: int, ece: float) -> Dict[str, float]:
    """Pack metrics into a flat dict.

    Args:
        conf: Segmentation confusion matrix.
        terrain_correct: Correct terrain predictions.
        terrain_total: Labeled terrain samples.
        ece: Calibration error.

    Returns:
        Dict with ``miou``, ``iou_drivable``, ``pixel_acc``, ``terrain_acc``, ``ece``.
    """
    ious = ious_from_confusion(conf)
    c = conf.double()
    return {"miou": float(np.nanmean(ious)), "iou_drivable": float(ious[1]),
            "pixel_acc": float((c.diag().sum() / c.sum().clamp(min=1)).item()),
            "terrain_acc": float(terrain_correct / terrain_total) if terrain_total else float("nan"),
            "ece": ece}
