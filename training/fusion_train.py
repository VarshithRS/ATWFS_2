"""FusionNet training script  --  STUB / PLACEHOLDER for when real training data exists.

!!! PLACEHOLDER !!! No real FusionNet training data exists yet. The expected CSV (``fusion.train_csv``) has columns
    uncertainty, geometric, temporal, terrain, target_alpha
How to create it later (suggestion, not implemented here): run the DL model and the classical fallback over
labelled frames; for each frame set ``target_alpha`` = 1 if IoU(DL, GT) > IoU(fallback, GT) else 0 (or a soft value
such as IoU_dl / (IoU_dl + IoU_fb)), and log the four input features computed exactly as in inference/live.py.
Until then the heuristic-initialised FusionNet (inference/fusion.py) is used in the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from inference.fusion import FusionNet
from utils.config import load_config, resolve_path
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)


def train_fusion(cfg: Dict[str, Any], csv_path: Optional[str] = None, epochs: int = 200, lr: float = 1e-2,
                 out_path: Optional[str] = None) -> Optional[str]:
    """Train FusionNet from a CSV (PLACEHOLDER pipeline; see module docstring).

    Args:
        cfg: Full config.
        csv_path: Training CSV path (default ``fusion.train_csv``).
        epochs: Full-batch epochs.
        lr: Learning rate.
        out_path: Output checkpoint path (default ``checkpoint_dir/fusion_net.pt``).

    Returns:
        Path of the saved checkpoint, or ``None`` if the CSV does not exist yet.
    """
    set_global_seed(cfg["seed"])
    p = Path(csv_path) if csv_path else resolve_path(cfg["fusion"]["train_csv"])
    if not p.exists():
        logger.warning("PLACEHOLDER: no FusionNet training data at %s. Keeping the hand-coded heuristic init.", p)
        return None
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    feats = torch.tensor([[float(r["uncertainty"]), float(r["geometric"]), float(r["temporal"])] for r in rows])
    terr = torch.tensor([int(r["terrain"]) for r in rows])
    y = torch.tensor([float(r["target_alpha"]) for r in rows])
    dev = get_device()
    net = FusionNet(cfg["fusion"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    feats, terr, y = feats.to(dev), terr.to(dev), y.to(dev)
    for ep in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy(net(feats, terr).clamp(1e-6, 1 - 1e-6), y)
        loss.backward(); opt.step()
        if ep % max(1, epochs // 5) == 0:
            logger.info("fusion epoch %d loss %.4f", ep, float(loss.detach()))
    out = Path(out_path) if out_path else resolve_path(cfg["paths"]["checkpoint_dir"]) / "fusion_net.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.cpu().state_dict(), out)
    logger.info("Saved FusionNet to %s (trained on %d rows)", out, len(rows))
    return str(out)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train FusionNet (stub until real data exists).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    setup_logging()
    train_fusion(load_config(a.config), a.csv)


if __name__ == "__main__":
    main()
