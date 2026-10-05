"""Grad-CAM for OFFLINE failure analysis (not part of the live pipeline).

Two targets:
  * ``terrain`` : class score of the terrain head, target layer = ``model.terrain_conv`` (mid-level features)
  * ``seg``     : mean (drivable - non-drivable) logit over the image, target layer = last encoder block
Typical use: run the failure taxonomy with ``--save-failures`` and point this script at the saved frames.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)


class GradCAM:
    """Minimal Grad-CAM using a forward hook + a tensor gradient hook (safe with in-place activations)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Attach to a layer.

        Args:
            model: Model.
            target_layer: Layer whose activations are explained.
        """
        self.model, self.act, self.grad = model, None, None
        self.handle = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, _m: nn.Module, _i: Any, out: torch.Tensor) -> None:
        self.act = out
        out.register_hook(lambda g: setattr(self, "grad", g))

    def close(self) -> None:
        """Remove the hook."""
        self.handle.remove()

    def __call__(self, x: torch.Tensor, task: str = "terrain", target_class: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Compute the heat-map.

        Args:
            x: ``(1,3,H,W)`` normalised image.
            task: ``terrain`` or ``seg``.
            target_class: Terrain class to explain (default: the predicted class).

        Returns:
            ``(cam (H,W) float in [0,1], explained class id)``.
        """
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            out = self.model(x)
            if task == "terrain":
                logits = out["terrain_logits"]
                cls = int(logits.argmax(1)) if target_class is None else int(target_class)
                score = logits[0, cls]
            else:
                cls = 1
                score = (out["seg_logits"][0, 1] - out["seg_logits"][0, 0]).mean()
            score.backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32), cls


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a JET heat-map over an RGB image.

    Args:
        rgb: ``(H, W, 3)`` uint8 RGB.
        cam: ``(H, W)`` float in [0, 1].
        alpha: Heat-map opacity.

    Returns:
        ``(H, W, 3)`` uint8 RGB overlay.
    """
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def explain_images(cfg: Dict[str, Any], model: nn.Module, images_dir: str, out_dir: str, task: str = "terrain",
                   device: Optional[torch.device] = None) -> List[str]:
    """Run Grad-CAM over every PNG/JPG in a folder and write overlays.

    Args:
        cfg: Full config.
        model: Model.
        images_dir: Folder with saved failure frames.
        out_dir: Output folder.
        task: ``terrain`` or ``seg``.
        device: Device.

    Returns:
        Paths of the written overlay files.
    """
    device = device or get_device()
    model = model.to(device).eval()
    layer = model.terrain_conv if task == "terrain" else model.encoder.features[-1]
    cam_fn = GradCAM(model, layer)
    h, w = cfg["data"]["image_size"]
    mean, std = np.asarray(cfg["data"]["imagenet_mean"], np.float32), np.asarray(cfg["data"]["imagenet_std"], np.float32)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outs: List[str] = []
    try:
        for p in sorted(Path(images_dir).glob("*.png")) + sorted(Path(images_dir).glob("*.jpg")):
            rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (w, h)), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(((rgb.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
            cam, cls = cam_fn(x, task)
            dst = Path(out_dir) / f"{p.stem}_{task}_cls{cls}.png"
            cv2.imwrite(str(dst), cv2.cvtColor(overlay(rgb, cam), cv2.COLOR_RGB2BGR))
            outs.append(str(dst))
    finally:
        cam_fn.close()
    logger.info("Wrote %d Grad-CAM overlays to %s", len(outs), out_dir)
    return outs


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Offline Grad-CAM on saved failure frames.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, help="folder of saved failure frames")
    ap.add_argument("--out", default="./outputs/reports/gradcam")
    ap.add_argument("--task", choices=["terrain", "seg"], default="terrain")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    set_global_seed(cfg["seed"])
    dev = get_device()
    model, _ = load_model_from_checkpoint(a.checkpoint, dev)
    explain_images(cfg, model, a.images, str(resolve_path(a.out)), a.task, dev)


if __name__ == "__main__":
    main()
