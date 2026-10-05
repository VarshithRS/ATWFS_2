"""Knowledge distillation: teacher (MobileNetV3-Large U-Net) -> student (MobileNetV3-Small, thinner decoder).

Total loss = ground-truth multi-task loss (student vs labels)
           + lambda * ( w_seg * KD_seg  +  w_conf * KD_conf  +  w_terrain * KD_terrain )
with soft-label matching of
  * segmentation  : T^2 * KL(softmax(t/T) || softmax(s/T)) per pixel
  * confidence    : MSE between the teacher's and student's evidential certainty (1 - K/S)
  * terrain       : T^2 * KL on terrain logits
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import build_dataloader, build_dataset
from models.unet_mobilenet import MultiTaskUNet, build_model, evidential_uncertainty
from training.losses import MultiTaskLoss
from training.train import ExtraLoss, load_model_from_checkpoint, run_training
from utils.config import load_config, resolve_path, smoke_config
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled KL divergence (teacher -> student), averaged over all non-class dims.

    Args:
        student_logits: ``(B,K,...)`` or ``(B,K)``.
        teacher_logits: Same shape.
        temperature: Softening temperature.

    Returns:
        Scalar loss.
    """
    t = temperature
    log_ps = F.log_softmax(student_logits / t, dim=1)
    pt = F.softmax(teacher_logits / t, dim=1)
    return (t * t) * (pt * (torch.log(pt.clamp_min(1e-8)) - log_ps)).sum(dim=1).mean()


def make_distill_loss(teacher: nn.Module, d_cfg: Dict[str, Any]) -> ExtraLoss:
    """Build the distillation term as a closure for :func:`training.train.run_training`.

    Args:
        teacher: Frozen teacher model.
        d_cfg: The ``distill`` config section.

    Returns:
        ``fn(batch, student_out) -> (loss, info)``.
    """
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    def fn(batch: Dict[str, torch.Tensor], s_out: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            t_out = teacher(batch["image"])
        t_out = {k: ([a.float() for a in v] if isinstance(v, list) else v.float()) for k, v in t_out.items()}
        kd_seg = kd_kl(s_out["seg_logits"], t_out["seg_logits"], d_cfg["temperature"])
        kd_conf = F.mse_loss(1 - evidential_uncertainty(s_out["alpha"]), 1 - evidential_uncertainty(t_out["alpha"]))
        kd_terr = kd_kl(s_out["terrain_logits"], t_out["terrain_logits"], d_cfg["temperature"])
        total = d_cfg["lambda_distill"] * (d_cfg["w_seg"] * kd_seg + d_cfg["w_conf"] * kd_conf + d_cfg["w_terrain"] * kd_terr)
        return total, {"kd_seg": float(kd_seg.detach()), "kd_conf": float(kd_conf.detach()), "kd_terrain": float(kd_terr.detach()), "kd_total": float(total.detach())}

    return fn


def load_teacher(cfg: Dict[str, Any], device: torch.device, ckpt: Optional[str] = None) -> MultiTaskUNet:
    """Load the teacher from a checkpoint (or random-init with a loud warning if absent).

    Args:
        cfg: Full config.
        device: Device.
        ckpt: Optional checkpoint path (defaults to ``distill.teacher_checkpoint``).

    Returns:
        Teacher model in eval mode.
    """
    path = Path(ckpt) if ckpt else resolve_path(cfg["distill"]["teacher_checkpoint"])
    if path.exists():
        model, _ = load_model_from_checkpoint(str(path), device)
        logger.info("Loaded teacher from %s", path)
        return model
    logger.warning("Teacher checkpoint %s not found -> using an UNTRAINED teacher (pipeline test only).", path)
    return build_model({**cfg["model"], "pretrained": False}, cfg["loss"]["evidence_clip"]).to(device).eval()


def train_distill(cfg: Dict[str, Any], synthetic: bool = False, teacher_ckpt: Optional[str] = None,
                  epochs: Optional[int] = None, tag: str = "student") -> Dict[str, Any]:
    """Distil the teacher into the student.

    Args:
        cfg: Full config.
        synthetic: Use synthetic data.
        teacher_ckpt: Teacher checkpoint path.
        epochs: Epoch override (default ``distill.epochs``).
        tag: Checkpoint prefix.

    Returns:
        Result dict from :func:`run_training`.
    """
    set_global_seed(cfg["seed"])
    device = get_device()
    teacher = load_teacher(cfg, device, teacher_ckpt)
    student = build_model(cfg["student"], cfg["loss"]["evidence_clip"])
    logger.info("Params: teacher %.2fM -> student %.2fM", sum(p.numel() for p in teacher.parameters()) / 1e6,
                sum(p.numel() for p in student.parameters()) / 1e6)
    tr = build_dataloader(build_dataset(cfg, "train", synthetic), cfg, True)
    va = build_dataloader(build_dataset(cfg, "val", synthetic), cfg, False)
    return run_training(cfg, student, MultiTaskLoss(cfg["loss"]), tr, va, device, tag, cfg["student"],
                        epochs or cfg["distill"]["epochs"], make_distill_loss(teacher.to(device), cfg["distill"]))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Knowledge distillation (teacher -> student).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    setup_logging()
    res = train_distill(load_config(a.config), a.synthetic, a.teacher, a.epochs)
    logger.info("Done. Student checkpoint: %s", res["best_path"])


if __name__ == "__main__":
    main()
