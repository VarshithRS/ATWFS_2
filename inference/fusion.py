"""ATWFS - Adaptive Trust-Weighted Fusion System.

``FusionNet`` maps (evidential uncertainty, geometric score, temporal score, terrain class) -> alpha in [0, 1],
the trust in the deep-learning mask. Final mask = alpha * DL_mask + (1 - alpha) * fallback_mask.

!!! PLACEHOLDER WARNING !!!
Real training data for FusionNet does not exist yet. The network is therefore INITIALISED with a
HAND-CODED HEURISTIC (see ``FusionNet.heuristic``) that it reproduces *exactly* at init, so the whole
pipeline works end-to-end today. Replace it by training with ``training/fusion_train.py`` once
(features, target_alpha) data has been collected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
NUM_TERRAIN = 4


class FusionNet(nn.Module):
    """Small MLP: 3 scalar features + one-hot terrain (7 inputs) -> hidden ReLU -> sigmoid alpha."""

    def __init__(self, fusion_cfg: Dict[str, Any], init_heuristic: bool = True) -> None:
        """Create the network.

        Args:
            fusion_cfg: The ``fusion`` config section.
            init_heuristic: Initialise weights to reproduce the hand-coded heuristic (PLACEHOLDER).
        """
        super().__init__()
        self.cfg = fusion_cfg
        hid = int(fusion_cfg["hidden"])
        self.fc1, self.fc2 = nn.Linear(3 + NUM_TERRAIN, hid), nn.Linear(hid, 1)
        if init_heuristic:
            self._init_from_heuristic()

    def _init_from_heuristic(self) -> None:
        """PLACEHOLDER init: encode ``z = gain * (w_c*(1-u) + w_g*geo + w_t*temp + bias - penalty[terrain])``.

        Uses the identity ``z = relu(z) - relu(-z)`` so a ReLU MLP reproduces the linear heuristic exactly.
        The remaining hidden units get tiny random weights so they can learn later.
        """
        c = self.cfg
        g = float(c["gain"])
        w = torch.zeros(3 + NUM_TERRAIN)
        w[0], w[1], w[2] = -g * c["w_certainty"], g * c["w_geometric"], g * c["w_temporal"]
        w[3:] = -g * torch.tensor(c["terrain_penalty"], dtype=torch.float32)
        b = g * (c["w_certainty"] + c["bias"])
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randn(self.fc1.weight.shape, generator=gen) * 1e-3)
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(torch.randn(self.fc2.weight.shape, generator=gen) * 1e-3)
            self.fc2.bias.zero_()
            self.fc1.weight[0], self.fc1.weight[1] = w, -w
            self.fc1.bias[0], self.fc1.bias[1] = b, -b
            self.fc2.weight[0, 0], self.fc2.weight[0, 1] = 1.0, -1.0

    @staticmethod
    def heuristic(cfg: Dict[str, Any], feats: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        """Reference hand-coded trust heuristic (PLACEHOLDER) the network is initialised to.

        Args:
            cfg: Fusion config.
            feats: ``(B,3)`` = (uncertainty, geometric, temporal).
            terrain: ``(B,)`` long terrain class.

        Returns:
            ``(B,)`` alpha in (0, 1).
        """
        pen = torch.tensor(cfg["terrain_penalty"], dtype=feats.dtype, device=feats.device)[terrain]
        z = cfg["gain"] * (cfg["w_certainty"] * (1 - feats[:, 0]) + cfg["w_geometric"] * feats[:, 1]
                           + cfg["w_temporal"] * feats[:, 2] + cfg["bias"] - pen)
        return torch.sigmoid(z)

    def forward(self, feats: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        """Compute alpha.

        Args:
            feats: ``(B,3)`` = (uncertainty, geometric score, temporal score), each in [0, 1].
            terrain: ``(B,)`` long terrain class ids.

        Returns:
            ``(B,)`` alpha in [0, 1].
        """
        x = torch.cat([feats.float(), F.one_hot(terrain.long().clamp(0, NUM_TERRAIN - 1), NUM_TERRAIN).float()], dim=1)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x)))).squeeze(1)


@dataclass
class FusionResult:
    """Output of :meth:`ATWFS.fuse`."""

    alpha: float            # trust in the DL mask, in [0, 1]
    soft_mask: np.ndarray   # float32 (H, W) in [0, 1]
    mask: np.ndarray        # uint8 (H, W) in {0, 1}
    active: str             # "DL" or "FALLBACK" (which estimator dominates)


class ATWFS:
    """Trust-weighted fusion of the DL mask and the classical fallback mask."""

    def __init__(self, fusion_cfg: Dict[str, Any], net: Optional[FusionNet] = None) -> None:
        """Create the fusion module.

        Args:
            fusion_cfg: The ``fusion`` config section.
            net: Optional pre-built (e.g. trained) FusionNet; defaults to heuristic-initialised placeholder.
        """
        self.cfg = fusion_cfg
        self.net = (net or FusionNet(fusion_cfg)).eval()

    @torch.no_grad()
    def alpha(self, uncertainty: float, geometric: float, temporal: float, terrain: int) -> float:
        """Compute the trust coefficient alpha.

        Args:
            uncertainty: Mean evidential uncertainty in [0, 1].
            geometric: Geometric plausibility score.
            temporal: Temporal consistency score.
            terrain: Terrain class id (0..3).

        Returns:
            alpha in [0, 1].
        """
        f = torch.tensor([[uncertainty, geometric, temporal]], dtype=torch.float32).clamp(0, 1)
        a = float(self.net(f, torch.tensor([terrain]))[0])
        return float(min(1.0, max(0.0, a)))

    def fuse(self, dl_mask: np.ndarray, fallback_mask: np.ndarray, uncertainty: float, geometric: float,
             temporal: float, terrain: int) -> FusionResult:
        """Fuse the two masks: ``alpha * DL + (1 - alpha) * fallback``.

        Args:
            dl_mask: ``(H, W)`` DL mask (binary or probability).
            fallback_mask: ``(H, W)`` classical mask (binary).
            uncertainty: Mean evidential uncertainty.
            geometric: Geometric score.
            temporal: Temporal score.
            terrain: Terrain class id.

        Returns:
            :class:`FusionResult`.
        """
        if dl_mask.shape != fallback_mask.shape:
            raise ValueError(f"mask shapes differ: {dl_mask.shape} vs {fallback_mask.shape}")
        a = self.alpha(uncertainty, geometric, temporal, terrain)
        soft = (a * dl_mask.astype(np.float32) + (1.0 - a) * fallback_mask.astype(np.float32)).astype(np.float32)
        return FusionResult(a, soft, (soft >= self.cfg["mask_threshold"]).astype(np.uint8),
                            "DL" if a >= self.cfg["fallback_threshold"] else "FALLBACK")
