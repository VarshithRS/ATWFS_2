"""Multi-task loss with Kendall-style learnable (homoscedastic) task weighting.

Tasks balanced by learnable log-variances ``s_i`` (loss_i * exp(-s_i) + s_i):
    seg      : CE + Dice + boundary-weighted CE  (+ aux_weight * auxiliary deep-supervision CE)
    evidence : Dirichlet / evidential loss (type-II ML + annealed KL regulariser)
    terrain  : cross-entropy (ignore_index = -1)
The auxiliary losses use a fixed weight (< 1) *inside* the seg task so they stay "lower than main".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def boundary_weight_map(mask: np.ndarray, w0: float = 5.0, sigma: float = 5.0) -> np.ndarray:
    """Distance-transform based edge weights: ``1 + w0 * exp(-d^2 / (2 sigma^2))``.

    ``d`` is the distance (px) to the nearest label boundary.

    Args:
        mask: ``(H, W)`` binary mask.
        w0: Extra weight right on the boundary.
        sigma: Spatial decay in pixels.

    Returns:
        ``(H, W)`` float32 weights (all ones if the mask has no boundary).
    """
    m = mask.astype(np.uint8)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if not edge.any():
        return np.ones(m.shape, np.float32)
    d = distance_transform_edt(~edge)
    return (1.0 + w0 * np.exp(-(d ** 2) / (2 * sigma ** 2))).astype(np.float32)


def batch_boundary_weights(masks: torch.Tensor, w0: float, sigma: float) -> torch.Tensor:
    """Boundary weights for a batch of masks.

    Args:
        masks: ``(B,H,W)`` long masks.
        w0: See :func:`boundary_weight_map`.
        sigma: See :func:`boundary_weight_map`.

    Returns:
        ``(B,H,W)`` float weights on the masks' device.
    """
    arr = masks.detach().cpu().numpy()
    w = np.stack([boundary_weight_map(a, w0, sigma) for a in arr])
    return torch.from_numpy(w).to(masks.device)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Multi-class soft Dice loss (mean over classes).

    Args:
        logits: ``(B,K,H,W)``.
        target: ``(B,H,W)`` long.
        eps: Smoothing.

    Returns:
        Scalar loss.
    """
    k = logits.shape[1]
    p = F.softmax(logits, dim=1)
    oh = F.one_hot(target, k).permute(0, 3, 1, 2).to(p.dtype)
    inter = (p * oh).sum(dim=(0, 2, 3))
    denom = p.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def evidential_loss(alpha: torch.Tensor, target: torch.Tensor, kl_coef: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evidential (Dirichlet) loss of Sensoy et al. 2018 (type-II maximum likelihood form).

    ``L = sum_k y_k (digamma(S) - digamma(alpha_k)) + kl_coef * KL(Dir(alpha~) || Dir(1))``
    where ``alpha~`` removes the evidence of the true class (penalises misleading evidence).

    Args:
        alpha: ``(B,K,H,W)`` Dirichlet parameters (>= 1).
        target: ``(B,H,W)`` long labels.
        kl_coef: Annealed regulariser coefficient in [0, 1].

    Returns:
        ``(total, fit_term)`` scalars.
    """
    k = alpha.shape[1]
    y = F.one_hot(target, k).permute(0, 3, 1, 2).to(alpha.dtype)
    s = alpha.sum(dim=1, keepdim=True)
    fit = (y * (torch.digamma(s) - torch.digamma(alpha))).sum(dim=1).mean()
    a_t = y + (1 - y) * alpha
    s_t = a_t.sum(dim=1, keepdim=True)
    kl = (torch.lgamma(s_t.squeeze(1)) - math_lgamma_k(k, alpha) - torch.lgamma(a_t).sum(dim=1)
          + ((a_t - 1) * (torch.digamma(a_t) - torch.digamma(s_t))).sum(dim=1)).mean()
    return fit + kl_coef * kl, fit


def math_lgamma_k(k: int, ref: torch.Tensor) -> torch.Tensor:
    """``lgamma(K)`` as a tensor on ``ref``'s device/dtype.

    Args:
        k: Number of classes.
        ref: Reference tensor.

    Returns:
        Scalar tensor.
    """
    return torch.lgamma(torch.tensor(float(k), device=ref.device, dtype=ref.dtype))


def terrain_ce(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """Cross-entropy that returns 0 (with a valid graph) when every label is ignored.

    Args:
        logits: ``(B,C)``.
        target: ``(B,)`` long, ``ignore_index`` marks unlabeled.
        ignore_index: Ignore value.

    Returns:
        Scalar loss.
    """
    valid = target != ignore_index
    ce = F.cross_entropy(logits, target.clamp(min=0), reduction="none")
    return (ce * valid).sum() / valid.sum().clamp(min=1)


class MultiTaskLoss(nn.Module):
    """Total loss for the multi-task network (see module docstring)."""

    TASKS = ("seg", "evidence", "terrain")

    def __init__(self, loss_cfg: Dict[str, Any]) -> None:
        """Create the loss.

        Args:
            loss_cfg: The ``loss`` config section.
        """
        super().__init__()
        self.cfg = loss_cfg
        self.log_vars = nn.Parameter(torch.zeros(len(self.TASKS)))  # learnable, Kendall et al.
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch (drives the evidential KL annealing).

        Args:
            epoch: 0-based epoch index.
        """
        self.epoch = epoch

    def forward(self, out: Dict[str, Any], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total loss.

        Args:
            out: Model output dict.
            batch: Batch with ``mask`` (B,H,W), ``terrain`` (B,), optional ``bweight`` (B,H,W).

        Returns:
            ``(total scalar, dict of detached component values)``.
        """
        c = self.cfg
        y = batch["mask"]
        logits = out["seg_logits"]
        bw = batch.get("bweight")
        if bw is None:
            bw = batch_boundary_weights(y, c["boundary"]["w0"], c["boundary"]["sigma"])
        ce_map = F.cross_entropy(logits, y, reduction="none")
        l_ce, l_dice = ce_map.mean(), dice_loss(logits, y)
        l_bnd = (ce_map * bw).sum() / bw.sum()
        l_aux = sum(F.cross_entropy(F.interpolate(a, size=y.shape[-2:], mode="bilinear", align_corners=False), y)
                    for a in out["aux_logits"]) / len(out["aux_logits"])
        l_seg = l_ce + c["dice_weight"] * l_dice + l_bnd + c["aux_weight"] * l_aux
        anneal = max(1, int(c["evidential_kl_anneal_epochs"]))
        kl_coef = min(1.0, self.epoch / anneal)
        l_evid, l_fit = evidential_loss(out["alpha"], y, kl_coef)
        l_terr = terrain_ce(out["terrain_logits"], batch["terrain"], c["terrain_ignore_index"])
        losses = torch.stack([l_seg, l_evid, l_terr])
        total = (torch.exp(-self.log_vars) * losses + self.log_vars).sum()
        info = {"loss": float(total.detach()), "seg": float(l_seg.detach()), "ce": float(l_ce.detach()),
                "dice": float(l_dice.detach()), "boundary": float(l_bnd.detach()), "aux": float(l_aux.detach()),
                "evidence": float(l_evid.detach()), "terrain": float(l_terr.detach()),
                **{f"w_{n}": float(torch.exp(-self.log_vars[i]).detach()) for i, n in enumerate(self.TASKS)}}
        return total, info
