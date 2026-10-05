"""Stage 8 smoke test: geometric + temporal scores on two consecutive synthetic frames."""
import math

import numpy as np

from data.synthetic import render_scene
from inference.consistency import geometric_score, temporal_score
from utils.config import load_config


def _gt_mask(vx: float) -> np.ndarray:
    _, raw, _ = render_scene(120, 160, np.random.default_rng(5), vx=vx, terrain=0, obstacle=0.5)
    return np.isin(raw, [0, 1]).astype(np.uint8)


def test_scores_finite_in_range_on_consecutive_frames() -> None:
    c = load_config()["consistency"]
    f1, f2 = _gt_mask(0.50), _gt_mask(0.52)
    g1, g2 = geometric_score(f1, c["geometric"]), geometric_score(f2, c["geometric"])
    t = temporal_score(f1, f2, 1.0, c["temporal"])
    for s in (g1, g2, t):
        assert isinstance(s, float) and math.isfinite(s) and 0.0 <= s <= 1.0
    assert g1 > 0.6, "a clean synthetic road should look plausible"
    assert t > 0.8, "tiny frame-to-frame shift should be consistent"


def test_temporal_behaviour() -> None:
    c = load_config()["consistency"]["temporal"]
    f1 = _gt_mask(0.30)
    f_far = _gt_mask(0.70)
    assert temporal_score(None, f1, 1.0, c) == 1.0
    assert temporal_score(f1, f1, 0.0, c) == 1.0
    slow, fast = temporal_score(f1, f_far, 0.0, c), temporal_score(f1, f_far, 10.0, c)
    assert slow < 1.0 and fast >= slow, "higher speed must tolerate larger shifts"
    assert temporal_score(f1, np.zeros_like(f1), 1.0, c) == 0.0


def test_geometric_penalises_bad_shapes() -> None:
    c = load_config()["consistency"]["geometric"]
    good = _gt_mask(0.5)
    assert geometric_score(np.zeros((64, 64), np.uint8), c) == 0.0
    rng = np.random.default_rng(0)
    speckle = (rng.random((120, 160)) > 0.5).astype(np.uint8)
    assert geometric_score(speckle, c) < geometric_score(good, c)
    inverted = good[::-1].copy()
    assert geometric_score(inverted, c) < geometric_score(good, c)
