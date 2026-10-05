"""Augmentation modules.

* :class:`StandardAugment`  - flip / colour jitter / random crop (image + masks).
* :class:`SyntheticConditionAugment` - rare-condition synthetic effects (fog, motion blur,
  water reflection, dust). Independent from the standard module and per-effect toggleable.

All operate on ``uint8`` RGB ``(H, W, 3)`` images and ``uint8`` ``(H, W)`` masks as numpy arrays
and never change shapes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class StandardAugment:
    """Horizontal flip, colour jitter and random-resized crop."""

    def __init__(self, cfg: Dict[str, Any], seed: int = 0) -> None:
        """Create the augmenter.

        Args:
            cfg: ``augmentation.standard`` config section.
            seed: RNG seed.
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        """Re-seed (used per DataLoader worker).

        Args:
            seed: New seed.
        """
        self.rng = np.random.default_rng(seed)

    def _jitter(self, img: np.ndarray) -> np.ndarray:
        c = self.cfg
        x = img.astype(np.float32)
        x *= 1.0 + self.rng.uniform(-c["brightness"], c["brightness"])
        mean = x.mean()
        x = (x - mean) * (1.0 + self.rng.uniform(-c["contrast"], c["contrast"])) + mean
        gray = x.mean(axis=2, keepdims=True)
        x = (x - gray) * (1.0 + self.rng.uniform(-c["saturation"], c["saturation"])) + gray
        x = np.clip(x, 0, 255).astype(np.uint8)
        if c["hue"] > 0:
            hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
            shift = int(self.rng.uniform(-c["hue"], c["hue"]) * 180)
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + shift) % 180).astype(np.uint8)
            x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return x

    def __call__(self, img: np.ndarray, masks: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Apply augmentation.

        Args:
            img: RGB uint8 image.
            masks: Masks that must receive the same geometric transform.

        Returns:
            Augmented ``(img, masks)`` with identical shapes to the input.
        """
        c = self.cfg
        if not c.get("enabled", True):
            return img, masks
        h, w = img.shape[:2]
        if self.rng.random() < c["hflip_p"]:
            img = img[:, ::-1].copy()
            masks = [m[:, ::-1].copy() for m in masks]
        if self.rng.random() < c["crop_p"]:
            s = self.rng.uniform(*c["crop_scale"])
            ch, cw = max(8, int(h * s)), max(8, int(w * s))
            y0 = int(self.rng.integers(0, h - ch + 1))
            x0 = int(self.rng.integers(0, w - cw + 1))
            img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
            masks = [cv2.resize(m[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]
        if self.rng.random() < c["jitter_p"]:
            img = self._jitter(img)
        return img, masks


class SyntheticConditionAugment:
    """Rare-condition image corruptions: fog, motion blur, water reflection, dust.

    The module can be disabled as a whole (``enabled``) or per effect.
    """

    def __init__(self, cfg: Dict[str, Any], seed: int = 0) -> None:
        """Create the augmenter.

        Args:
            cfg: ``augmentation.synthetic`` config section.
            seed: RNG seed.
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        """Re-seed the RNG.

        Args:
            seed: New seed.
        """
        self.rng = np.random.default_rng(seed)

    # -- individual effects (also callable directly, e.g. in tests) ----------------
    def fog(self, img: np.ndarray, density: float) -> np.ndarray:
        """Blend a depth-like haze (denser toward the horizon).

        Args:
            img: RGB uint8 image.
            density: Haze strength in [0, 1].

        Returns:
            Fogged image.
        """
        h, w = img.shape[:2]
        ramp = np.linspace(1.0, 0.35, h, dtype=np.float32)[:, None, None]  # denser at top
        noise = cv2.GaussianBlur(self.rng.random((h, w)).astype(np.float32), (0, 0), max(h, w) / 10.0)
        noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)
        a = np.clip(density * ramp * (0.7 + 0.3 * noise[..., None]), 0, 0.95)
        haze = np.full_like(img, 215, dtype=np.uint8).astype(np.float32)
        return np.clip(img.astype(np.float32) * (1 - a) + haze * a, 0, 255).astype(np.uint8)

    def motion_blur(self, img: np.ndarray, ksize: int) -> np.ndarray:
        """Directional linear blur.

        Args:
            img: RGB uint8 image.
            ksize: Kernel length in pixels (>=3).

        Returns:
            Blurred image.
        """
        ksize = max(3, int(ksize) | 1)
        kernel = np.zeros((ksize, ksize), np.float32)
        kernel[ksize // 2, :] = 1.0
        angle = float(self.rng.uniform(-30, 30))
        rot = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
        kernel /= kernel.sum() + 1e-6
        return cv2.filter2D(img, -1, kernel)

    def water_reflection(self, img: np.ndarray, strength: float) -> np.ndarray:
        """Paint puddle-like patches that show a wavy, vertically-flipped reflection.

        Args:
            img: RGB uint8 image.
            strength: Blend factor of the reflection in [0, 1].

        Returns:
            Image with reflective patches in its lower half.
        """
        h, w = img.shape[:2]
        refl = img[::-1].copy()
        xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
        ys = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
        xs += 3.0 * np.sin(ys / 4.0)
        refl = cv2.remap(refl, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        patch = np.zeros((h, w), np.float32)
        for _ in range(int(self.rng.integers(1, 4))):
            cx = int(self.rng.integers(0, w))
            cy = int(self.rng.integers(int(h * 0.55), h))
            ax, ay = int(self.rng.integers(max(2, w // 10), max(3, w // 4))), int(self.rng.integers(max(2, h // 20), max(3, h // 8)))
            cv2.ellipse(patch, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
        patch = cv2.GaussianBlur(patch, (0, 0), 2.0)[..., None] * strength
        return np.clip(img.astype(np.float32) * (1 - patch) + refl.astype(np.float32) * patch, 0, 255).astype(np.uint8)

    def dust(self, img: np.ndarray, density: float) -> np.ndarray:
        """Tan haze plus speckle noise.

        Args:
            img: RGB uint8 image.
            density: Strength in [0, 1].

        Returns:
            Dusty image.
        """
        h, w = img.shape[:2]
        tint = np.array([190, 160, 115], np.float32)
        speck = (self.rng.random((h, w, 1)).astype(np.float32) > 0.985) * 60.0
        out = img.astype(np.float32) * (1 - density * 0.6) + tint * density * 0.6 + speck * density
        return np.clip(cv2.GaussianBlur(out, (3, 3), 0), 0, 255).astype(np.uint8)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Randomly apply each enabled effect with its own probability.

        Args:
            img: RGB uint8 image.

        Returns:
            Image of identical shape.
        """
        c = self.cfg
        if not c.get("enabled", False):
            return img
        fx = c["fog"]
        if fx.get("enabled") and self.rng.random() < fx["p"]:
            img = self.fog(img, float(self.rng.uniform(0.2, fx["max_density"])))
        fx = c["motion_blur"]
        if fx.get("enabled") and self.rng.random() < fx["p"]:
            img = self.motion_blur(img, int(self.rng.integers(5, fx["max_kernel"] + 1)))
        fx = c["water_reflection"]
        if fx.get("enabled") and self.rng.random() < fx["p"]:
            img = self.water_reflection(img, fx["strength"])
        fx = c["dust"]
        if fx.get("enabled") and self.rng.random() < fx["p"]:
            img = self.dust(img, float(self.rng.uniform(0.15, fx["max_density"])))
        return img
