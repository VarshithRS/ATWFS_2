"""Stage 7 smoke test: classical estimator returns a valid binary mask of the right shape."""
import cv2
import numpy as np

from data.synthetic import render_scene
from inference.classical import ClassicalFreeSpaceEstimator
from utils.config import load_config


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def test_classical_mask_valid_and_sane() -> None:
    est = ClassicalFreeSpaceEstimator(load_config()["classical"])
    ious = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        img, raw, _ = render_scene(120, 160, rng, vx=0.5, terrain=0, obstacle=None)
        m = est.estimate(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        assert m.shape == (120, 160) and m.dtype == np.uint8
        assert set(np.unique(m).tolist()) <= {0, 1}
        ious.append(_iou(m.astype(bool), np.isin(raw, [0, 1])))
    assert max(ious) > 0.3, f"classical estimator finds almost no road: {ious}"


def test_classical_on_blank_and_noise_does_not_crash() -> None:
    est = ClassicalFreeSpaceEstimator(load_config()["classical"])
    for img in (np.zeros((64, 96, 3), np.uint8), np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8)):
        m = est.estimate(img)
        assert m.shape == (64, 96) and set(np.unique(m).tolist()) <= {0, 1}
