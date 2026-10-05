"""Obstacle detection for obstacle subtraction.

Default: pretrained COCO YOLOv8n via the ``ultralytics`` package, filtered to person/vehicle classes.

=====================  CATTLE-DETECTOR PLUG-IN POINT  =====================
To add a custom cattle detector later:
  1. fine-tune a YOLO model on your cattle data (COCO already has "cow" = id 19 but it is weak for Indian
     breeds / roadside herds),
  2. implement the :class:`ObstacleDetector` interface (``detect(bgr) -> List[Detection]``) - or simply point
     ``inference.detector.weights`` at the fine-tuned weights and add its class ids to ``class_ids`` in config.yaml,
  3. register it in :func:`build_detector` by wrapping both detectors in :class:`CompositeDetector`.
:class:`CattleDetectorPlaceholder` below marks the spot; it currently detects nothing (STUB).
============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from utils.config import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected obstacle (pixel coordinates in the input frame)."""

    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    name: str
    conf: float


class ObstacleDetector:
    """Interface for obstacle detectors."""

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        """Detect obstacles.

        Args:
            bgr: ``(H, W, 3)`` uint8 BGR frame.

        Returns:
            List of detections.
        """
        raise NotImplementedError


class NullDetector(ObstacleDetector):
    """Detects nothing (used when YOLO is disabled/unavailable)."""

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        """Return no detections.

        Args:
            bgr: Frame (ignored).

        Returns:
            Empty list.
        """
        return []


class YoloV8nDetector(ObstacleDetector):
    """Pretrained YOLOv8n filtered to person/vehicle classes (see config ``inference.detector``)."""

    def __init__(self, weights: str, conf: float, class_ids: Sequence[int]) -> None:
        """Load the model.

        Args:
            weights: Weights path / name.
            conf: Confidence threshold.
            class_ids: COCO class ids to keep.
        """
        from ultralytics import YOLO  # imported lazily: optional dependency

        self.model = YOLO(weights)
        self.conf, self.class_ids = conf, list(class_ids)
        self.names: Dict[int, str] = dict(self.model.names)

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        """Run YOLO on a frame.

        Args:
            bgr: BGR frame.

        Returns:
            Filtered detections.
        """
        res = self.model.predict(bgr, conf=self.conf, classes=self.class_ids, verbose=False)[0]
        out: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            out.append(Detection(int(x1), int(y1), int(x2), int(y2), int(c), self.names.get(int(c), str(int(c))), float(p)))
        return out


class CattleDetectorPlaceholder(ObstacleDetector):
    """PLACEHOLDER / STUB for a future fine-tuned cattle detector (currently detects nothing)."""

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        """Return no detections (stub).

        Args:
            bgr: Frame (ignored).

        Returns:
            Empty list.
        """
        return []


class CompositeDetector(ObstacleDetector):
    """Union of several detectors (e.g. YOLOv8n + a future cattle detector)."""

    def __init__(self, detectors: Sequence[ObstacleDetector]) -> None:
        """Create the composite.

        Args:
            detectors: Detectors to run.
        """
        self.detectors = list(detectors)

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        """Run all detectors and concatenate results.

        Args:
            bgr: BGR frame.

        Returns:
            Combined detections.
        """
        return [d for det in self.detectors for d in det.detect(bgr)]


def build_detector(det_cfg: Dict[str, Any], output_dir: str) -> ObstacleDetector:
    """Build the configured detector; degrade gracefully to :class:`NullDetector` on any failure.

    Args:
        det_cfg: ``inference.detector`` config section.
        output_dir: ``paths.output_dir`` (weights are cached in ``<output_dir>/weights``).

    Returns:
        An :class:`ObstacleDetector`.
    """
    if det_cfg["backend"] == "none":
        return NullDetector()
    try:
        wdir = resolve_path(output_dir) / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        wpath = wdir / Path(det_cfg["weights"]).name
        det: ObstacleDetector = YoloV8nDetector(str(wpath), det_cfg["conf"], det_cfg["class_ids"])
        logger.info("Obstacle detector: YOLO (%s), classes %s", wpath.name, det_cfg["class_ids"])
        # CATTLE PLUG-IN: return CompositeDetector([det, YourCattleDetector(...)])
        return det
    except Exception as e:
        logger.warning("YOLO obstacle detector unavailable (%s: %s). Continuing WITHOUT obstacle detection - "
                       "install `ultralytics` and ensure the weights can be downloaded.", type(e).__name__, str(e)[:120])
        return NullDetector()
