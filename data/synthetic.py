"""Procedural fake road scenes (images + level-3 id masks + terrain labels).

No real data needed: lets every stage be smoke-tested on a CPU-only machine.
Masks use the *real* IDD level-3 id space so the label-collapsing code is exercised
end to end (road=0, drivable fallback=1, sidewalk=2, non-drivable fallback=3,
person=4, car=9, building=22, vegetation=24, sky=25).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# terrain class -> road surface RGB colour
_TERRAIN_COLORS = {0: (105, 105, 110), 1: (176, 140, 98), 2: (92, 128, 70), 3: (80, 110, 150)}


def render_scene(h: int, w: int, rng: np.random.Generator, vx: Optional[float] = None,
                 terrain: Optional[int] = None, obstacle: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Render one synthetic scene.

    Args:
        h: Height in px.
        w: Width in px.
        rng: Random generator (controls all unspecified params).
        vx: Optional vanishing-point x (fraction of width, 0..1).
        terrain: Optional terrain class 0..3 for the road surface.
        obstacle: Optional obstacle depth position in [0, 1] (0 = far, 1 = near); None = random/no obstacle.

    Returns:
        ``(image RGB uint8 (h,w,3), level3-id mask uint8 (h,w), terrain class)``.
    """
    horizon = int(h * rng.uniform(0.30, 0.42))
    vx_px = int((vx if vx is not None else rng.uniform(0.3, 0.7)) * w)
    terrain = int(rng.integers(0, 4)) if terrain is None else int(terrain)
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.full((h, w), 255, np.uint8)

    def fill(poly: np.ndarray, color: Tuple[int, int, int], cid: int) -> None:
        cv2.fillPoly(img, [poly.astype(np.int32)], color)
        cv2.fillPoly(mask, [poly.astype(np.int32)], int(cid))

    img[:horizon] = (135, 180, 225)
    mask[:horizon] = 25
    ground = (120, 150, 90) if terrain != 2 else (150, 160, 100)
    img[horizon:] = ground
    mask[horizon:] = 3
    # buildings / vegetation along the horizon
    for _ in range(int(rng.integers(2, 5))):
        bx = int(rng.integers(0, w))
        bw, bh = int(rng.integers(w // 10, w // 4)), int(rng.integers(h // 12, h // 5))
        col, cid = ((150, 140, 130), 22) if rng.random() < 0.5 else ((60, 110, 50), 24)
        fill(np.array([[bx, horizon], [bx + bw, horizon], [bx + bw, horizon - bh], [bx, horizon - bh]]), col, cid)

    bw_bottom, bw_top = w * rng.uniform(0.28, 0.42), w * rng.uniform(0.02, 0.06)
    xb = vx_px + (0.5 * w - vx_px) * 0.6  # road base drifts toward the centre
    def road_poly(scale: float) -> np.ndarray:
        return np.array([[xb - bw_bottom * scale, h], [xb + bw_bottom * scale, h],
                         [vx_px + bw_top * scale, horizon], [vx_px - bw_top * scale, horizon]])
    fill(road_poly(1.45), (170, 170, 165), 2)                # sidewalk
    if rng.random() < 0.7:
        fill(road_poly(1.15), (140, 120, 135), 1)            # drivable fallback / shoulder
    fill(road_poly(1.0), _TERRAIN_COLORS[terrain], 0)        # road

    n_obs = 1 if (obstacle is not None or rng.random() < 0.6) else 0
    for _ in range(n_obs):
        depth = obstacle if obstacle is not None else rng.uniform(0.2, 0.9)
        y1 = int(horizon + depth * (h - horizon))
        ow = max(3, int(w * (0.04 + 0.12 * depth)))
        oh = max(3, int(h * (0.04 + 0.12 * depth)))
        ox = int(vx_px + (xb - vx_px) * depth + rng.uniform(-0.15, 0.15) * bw_bottom * depth)
        if rng.random() < 0.5:
            fill(np.array([[ox - ow, y1], [ox + ow, y1], [ox + ow, y1 - oh], [ox - ow, y1 - oh]]), (200, 30, 30), 9)
        else:
            fill(np.array([[ox - ow // 3, y1], [ox + ow // 3, y1], [ox + ow // 3, y1 - oh * 2], [ox - ow // 3, y1 - oh * 2]]), (230, 200, 40), 4)

    noise = rng.normal(0, 6, img.shape).astype(np.float32)
    img = np.clip(cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32) + noise, 0, 255).astype(np.uint8)
    mask[mask == 255] = 3
    return img, mask, terrain


class SyntheticSource:
    """Sample source producing ``(img, raw_mask, terrain)`` procedurally."""

    def __init__(self, num_samples: int, image_size: Tuple[int, int], seed: int = 0) -> None:
        """Create a source.

        Args:
            num_samples: Dataset length.
            image_size: ``(H, W)``.
            seed: Base seed (different seeds -> disjoint train/val sets).
        """
        self.n, self.size, self.seed = num_samples, tuple(image_size), seed

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n

    def load(self, i: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Render sample ``i`` deterministically.

        Args:
            i: Index.

        Returns:
            ``(image, raw level3 mask, terrain class)``.
        """
        rng = np.random.default_rng(self.seed * 100_003 + i)
        return render_scene(self.size[0], self.size[1], rng)


def write_synthetic_video(path: str, num_frames: int = 12, size_wh: Tuple[int, int] = (160, 120),
                          fps: int = 10, seed: int = 0, fourcc: str = "mp4v") -> str:
    """Write a short synthetic driving clip with a smoothly drifting road and an approaching obstacle.

    Args:
        path: Output path (extension may change to ``.avi`` if the mp4 codec is unavailable).
        num_frames: Number of frames.
        size_wh: ``(W, H)``.
        fps: Frame rate.
        seed: Seed.
        fourcc: Preferred codec.

    Returns:
        The path actually written.
    """
    w, h = size_wh
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    if not writer.isOpened():
        path = str(Path(path).with_suffix(".avi"))
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    terrain = int(np.random.default_rng(seed).integers(0, 2))
    for k in range(num_frames):
        rng = np.random.default_rng(seed * 7919 + 1)  # same scene layout every frame ...
        t = k / max(1, num_frames - 1)
        img, _, _ = render_scene(h, w, rng, vx=0.45 + 0.1 * np.sin(2 * np.pi * t), terrain=terrain,
                                 obstacle=0.3 + 0.5 * t)  # ... only vx and obstacle move
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    logger.info("Wrote synthetic video %s (%d frames)", path, num_frames)
    return path
