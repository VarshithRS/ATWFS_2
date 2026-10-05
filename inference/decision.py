"""Steering + speed decision logic and obstacle subtraction.

Formulas (all constants live in ``config.yaml -> inference.decision``):

  look-ahead band  : rows  [r0*H, r1*H)  of the final (obstacle-subtracted) drivable mask
  centroid x (cx)  : mean column of drivable pixels in the band
  angle (deg)      : max_steer_deg * clip((cx - W/2) / (W/2), -1, 1)           (+ = steer right)
  width_frac       : median over band rows of (#drivable pixels in row / W)

  obstacle         : a detection box is "in the corridor" if it overlaps [cx - ch*W, cx + ch*W]
                     proximity = max over in-corridor boxes of (box_bottom_y / H)   (larger = closer)
                     obstacle_flag = proximity >= prox_slow
                     obs_factor    = clip((prox_stop - proximity) / (prox_stop - prox_slow), 0, 1)  (1 if none)

  speed (m/s)      = v_max * trust_factor * width_factor * obs_factor * terrain_factor
      trust_factor   = clip((trust - trust_min) / (1 - trust_min), 0, 1)
      width_factor   = clip(width_frac / width_ref, 0, 1)
      terrain_factor = terrain_speed_factor[terrain_class]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from inference.detector import Detection


@dataclass
class Command:
    """Decision output; ``packet`` is the [angle, speed, trust, obstacle_flag] STUB for a future Arduino link."""

    angle: float
    speed: float
    trust: float
    obstacle_flag: bool
    width_frac: float
    centroid_x: float
    obstacle_proximity: float

    @property
    def packet(self) -> Tuple[float, float, float, int]:
        """The ``[angle, speed, trust_score, obstacle_flag]`` packet (NO serial/hardware code exists in this project)."""
        return (self.angle, self.speed, self.trust, int(self.obstacle_flag))


def subtract_obstacles(mask: np.ndarray, dets: Sequence[Detection], dilate_px: int) -> np.ndarray:
    """Remove detected-obstacle boxes (expanded by ``dilate_px``) from the drivable mask.

    Args:
        mask: ``(H, W)`` binary mask.
        dets: Detections in the same pixel frame.
        dilate_px: Box expansion in pixels.

    Returns:
        New mask with obstacle regions set to 0.
    """
    out = mask.copy()
    h, w = out.shape
    for d in dets:
        out[max(0, d.y1 - dilate_px):min(h, d.y2 + dilate_px + 1), max(0, d.x1 - dilate_px):min(w, d.x2 + dilate_px + 1)] = 0
    return out


def drivable_geometry(mask: np.ndarray, band: Sequence[float]) -> Tuple[float, float]:
    """Centroid column and drivable width in the look-ahead band.

    Args:
        mask: ``(H, W)`` binary mask.
        band: ``(r0, r1)`` row fractions of the look-ahead band.

    Returns:
        ``(centroid_x in px, width_frac)``; centroid defaults to the image centre and width to 0 if the band is empty.
    """
    h, w = mask.shape
    sub = mask[int(band[0] * h):max(int(band[0] * h) + 1, int(band[1] * h))] > 0
    if not sub.any():
        return w / 2.0, 0.0
    cols = np.nonzero(sub)[1]
    return float(cols.mean()), float(np.median(sub.sum(axis=1)) / w)


def assess_obstacles(dets: Sequence[Detection], frame_shape: Tuple[int, int], centroid_x: float,
                     cfg: Dict[str, Any]) -> Tuple[bool, float, float]:
    """Corridor obstacle check.

    Args:
        dets: Detections.
        frame_shape: ``(H, W)``.
        centroid_x: Steering centre column.
        cfg: ``inference.decision`` config.

    Returns:
        ``(obstacle_flag, proximity in [0,1], obs_factor in [0,1])``.
    """
    h, w = frame_shape
    lo, hi = centroid_x - cfg["corridor_half_width"] * w, centroid_x + cfg["corridor_half_width"] * w
    prox = 0.0
    for d in dets:
        if d.x2 >= lo and d.x1 <= hi:
            prox = max(prox, min(1.0, d.y2 / float(h)))
    if prox == 0.0:
        return False, 0.0, 1.0
    factor = float(np.clip((cfg["prox_stop"] - prox) / max(1e-6, cfg["prox_stop"] - cfg["prox_slow"]), 0.0, 1.0))
    return prox >= cfg["prox_slow"], prox, factor


def compute_command(final_mask: np.ndarray, dets: Sequence[Detection], trust: float, terrain: int,
                    cfg: Dict[str, Any]) -> Command:
    """Compute steering angle and speed (formulas in the module docstring).

    Args:
        final_mask: Obstacle-subtracted binary mask at frame resolution.
        dets: Detections at frame resolution.
        trust: Trust score (alpha) in [0, 1].
        terrain: Terrain class id.
        cfg: ``inference.decision`` config.

    Returns:
        :class:`Command`.
    """
    h, w = final_mask.shape
    cx, width_frac = drivable_geometry(final_mask, cfg["lookahead_rows"])
    angle = float(cfg["max_steer_deg"] * np.clip((cx - w / 2.0) / (w / 2.0), -1.0, 1.0)) if width_frac > 0 else 0.0
    flag, prox, obs_f = assess_obstacles(dets, (h, w), cx, cfg)
    trust_f = float(np.clip((trust - cfg["trust_min"]) / max(1e-6, 1.0 - cfg["trust_min"]), 0.0, 1.0))
    width_f = float(np.clip(width_frac / cfg["width_ref"], 0.0, 1.0))
    terr_f = float(cfg["terrain_speed_factor"][int(np.clip(terrain, 0, len(cfg["terrain_speed_factor"]) - 1))])
    speed = float(cfg["v_max"] * trust_f * width_f * obs_f * terr_f)
    return Command(angle, speed, float(trust), bool(flag), width_frac, cx, prox)
