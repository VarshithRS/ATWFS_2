"""Classical (non-neural) free-space estimator: colour-space seeding + edge barriers (OpenCV).

Idea: assume the patch directly in front of the vehicle (a trapezoid at the bottom-centre of the frame)
is drivable. Learn its colour statistics in CIE-Lab, mark pixels with similar colour as candidate free
space, cut candidates by strong Canny edges (obstacle/road boundaries), then keep the connected
region(s) that touch the seed and clean up with morphology.
This is the FALLBACK used when the learned model is untrusted (see :mod:`inference.fusion`).
"""
from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


class ClassicalFreeSpaceEstimator:
    """Colour + edge heuristic free-space estimator."""

    _SIGMA_MIN = np.array([8.0, 4.0, 4.0], np.float32)  # floors for (L, a, b) std -> avoids over-tight models

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Create the estimator.

        Args:
            cfg: The ``classical`` config section.
        """
        self.cfg = cfg

    def seed_mask(self, h: int, w: int) -> np.ndarray:
        """Trapezoidal seed region in front of the vehicle.

        Args:
            h: Image height.
            w: Image width.

        Returns:
            ``(h, w)`` uint8 {0,1} seed mask.
        """
        s = self.cfg["seed_region"]
        pts = np.array([[w / 2 - s["half_width_top"] * w, s["top"] * h], [w / 2 + s["half_width_top"] * w, s["top"] * h],
                        [w / 2 + s["half_width_bottom"] * w, s["bottom"] * h], [w / 2 - s["half_width_bottom"] * w, s["bottom"] * h]])
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts.astype(np.int32)], 1)
        return m

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        """Estimate the free-space mask.

        Args:
            bgr: ``(H, W, 3)`` uint8 BGR frame.

        Returns:
            ``(H, W)`` uint8 binary mask (1 = free/drivable); may be all zeros if nothing plausible is found.
        """
        c = self.cfg
        h, w = bgr.shape[:2]
        blur = cv2.GaussianBlur(bgr, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        seed = self.seed_mask(h, w).astype(bool)
        pix = lab[seed]
        mu, sd = pix.mean(0), np.maximum(pix.std(0), self._SIGMA_MIN)
        dist = np.sqrt((((lab - mu) / sd) ** 2).mean(axis=2))
        free = dist < c["color_k"]
        edges = cv2.Canny(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), *c["canny"])
        if c["edge_dilate"] > 0:
            edges = cv2.dilate(edges, np.ones((2 * c["edge_dilate"] + 1,) * 2, np.uint8))
        free &= edges == 0
        free[: int(c["horizon_frac"] * h)] = False
        k = np.ones((c["morph_kernel"],) * 2, np.uint8)
        free_u8 = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, k)
        free_u8 = cv2.morphologyEx(free_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lbl = cv2.connectedComponents(free_u8, connectivity=8)
        if n <= 1:
            return np.zeros((h, w), np.uint8)
        overlap = np.bincount(lbl[seed].ravel(), minlength=n)
        overlap[0] = 0
        keep = np.flatnonzero(overlap > 0.05 * seed.sum())
        if keep.size == 0:
            return np.zeros((h, w), np.uint8)
        out = np.isin(lbl, keep).astype(np.uint8)
        # fill interior holes (e.g. lane-marking edges, small stones) by flood-filling the background
        inv = (1 - out).astype(np.uint8)
        ff = inv.copy()
        cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
        holes = (ff == 1)
        out[holes] = 1
        return out
