"""Geometric and temporal consistency scores for drivable-region masks (both in [0, 1], 1 = plausible)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _clip01(x: float) -> float:
    """Clamp to [0, 1] and guarantee a finite float.

    Args:
        x: Value.

    Returns:
        Clamped float (0.0 if non-finite).
    """
    return float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else 0.0


def geometric_score(mask: np.ndarray, cfg: Dict[str, Any]) -> float:
    """Contour-based road-shape plausibility.

    Sub-scores (weights from ``consistency.geometric.weights``):
      * area         - drivable area fraction within a plausible range
      * solidity     - area / convex-hull area of the main region (roads are roughly convex)
      * perspective  - region widens toward the bottom (corr of row index with row width)
      * connectivity - fraction of the area in the largest connected region (no fragments)
      * bottom_touch - region reaches the bottom of the frame (the road under the vehicle)

    Args:
        mask: ``(H, W)`` binary mask.
        cfg: The ``consistency.geometric`` config section.

    Returns:
        Plausibility score in [0, 1]; 0.0 for an empty mask.
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    total = int(m.sum())
    if total == 0:
        return 0.0
    lo, hi = cfg["area_frac_range"]
    frac = total / float(h * w)
    s_area = frac / lo if frac < lo else (1.0 if frac <= hi else (1.0 - frac) / max(1e-6, 1.0 - hi))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    big = contours[int(np.argmax(areas))]
    hull_area = cv2.contourArea(cv2.convexHull(big))
    solidity = (max(areas) / hull_area) if hull_area > 0 else 0.0
    s_solid = (solidity - 0.4) / 0.6
    s_conn = max(areas) / max(1.0, float(sum(areas)))
    comp = np.zeros_like(m)
    cv2.drawContours(comp, [big], -1, 1, thickness=-1)
    rows = np.flatnonzero(comp.any(axis=1))
    widths = comp[rows].sum(axis=1).astype(np.float64)
    if rows.size >= 5 and widths.std() > 1e-6:
        s_persp = 0.5 * (np.corrcoef(rows, widths)[0, 1] + 1.0)
    else:
        s_persp = 0.5
    band = m[int(0.92 * h):]
    s_bottom = float(band.mean()) / 0.15
    wts = cfg["weights"]
    parts = {"area": s_area, "solidity": s_solid, "perspective": s_persp, "connectivity": s_conn, "bottom_touch": s_bottom}
    score = sum(wts[k] * _clip01(v) for k, v in parts.items()) / sum(wts[k] for k in parts)
    return _clip01(score)


def _row_edges(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left/right-most drivable column per row.

    Args:
        m: ``(H, W)`` bool mask.

    Returns:
        ``(has_pixels (H,), left (H,), right (H,))``.
    """
    has = m.any(axis=1)
    left = m.argmax(axis=1)
    right = m.shape[1] - 1 - m[:, ::-1].argmax(axis=1)
    return has, left, right


def temporal_score(prev_mask: Optional[np.ndarray], curr_mask: np.ndarray, speed_mps: float,
                   cfg: Dict[str, Any]) -> float:
    """Frame-to-frame boundary-shift consistency, normalised by vehicle speed.

    The boundary shift is the mean absolute displacement of the left and right drivable boundaries over rows
    where both masks have pixels, as a fraction of image width. The tolerated shift grows with speed:
    ``tol = base_shift + expected_shift_per_mps * speed * dt``. Score = ``exp(-max(0, shift - tol) / tau)``.

    Args:
        prev_mask: Previous ``(H, W)`` binary mask or ``None`` (first frame -> 1.0, i.e. no evidence of inconsistency).
        curr_mask: Current ``(H, W)`` binary mask.
        speed_mps: Vehicle speed input (m/s, >= 0).
        cfg: The ``consistency.temporal`` config section.

    Returns:
        Score in (0, 1]; 0.0 if exactly one of the masks is empty.
    """
    if prev_mask is None:
        return 1.0
    a, b = prev_mask > 0, curr_mask > 0
    if not a.any() and not b.any():
        return 1.0
    if a.any() != b.any():
        return 0.0
    ha, la, ra = _row_edges(a)
    hb, lb, rb = _row_edges(b)
    common = ha & hb
    if not common.any():
        return 0.0
    w = a.shape[1]
    shift = float(np.mean((np.abs(la[common] - lb[common]) + np.abs(ra[common] - rb[common])) / 2.0)) / w
    tol = cfg["base_shift"] + cfg["expected_shift_per_mps"] * max(0.0, float(speed_mps)) * cfg["dt"]
    return _clip01(float(np.exp(-max(0.0, shift - tol) / cfg["tau"])))
