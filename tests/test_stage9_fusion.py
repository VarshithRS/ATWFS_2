"""Stage 9 smoke test: alpha in [0,1], mask shapes, heuristic init reproduced, training stub runs."""
import csv

import numpy as np
import torch

from inference.fusion import ATWFS, FusionNet
from training.fusion_train import train_fusion
from utils.config import resolve_path, smoke_config


def test_alpha_always_in_unit_interval_and_shapes() -> None:
    cfg = smoke_config()["fusion"]
    f = ATWFS(cfg)
    rng = np.random.default_rng(0)
    dl = (rng.random((48, 64)) > 0.5).astype(np.uint8)
    fb = (rng.random((48, 64)) > 0.5).astype(np.uint8)
    for _ in range(200):
        u, g, t = rng.random(3)
        res = f.fuse(dl, fb, u, g, t, int(rng.integers(0, 4)))
        assert 0.0 <= res.alpha <= 1.0
        assert res.mask.shape == (48, 64) and res.soft_mask.shape == (48, 64)
        assert set(np.unique(res.mask).tolist()) <= {0, 1}
        assert res.soft_mask.min() >= 0 and res.soft_mask.max() <= 1
    # extremes (even out-of-range inputs are clamped)
    for feats in [(0, 1, 1), (1, 0, 0), (5, -3, 9)]:
        assert 0.0 <= f.alpha(*feats, 0) <= 1.0


def test_heuristic_init_is_reproduced_and_sensible() -> None:
    cfg = smoke_config()["fusion"]
    net = FusionNet(cfg).eval()
    feats = torch.rand(64, 3)
    terr = torch.randint(0, 4, (64,))
    with torch.no_grad():
        assert torch.allclose(net(feats, terr), FusionNet.heuristic(cfg, feats, terr), atol=1e-3)
    f = ATWFS(cfg)
    assert f.alpha(0.05, 0.95, 0.95, 0) > 0.9 > 0.2 > f.alpha(0.95, 0.3, 0.3, 3)


def test_fusion_blend_and_active_label() -> None:
    f = ATWFS(smoke_config()["fusion"])
    ones, zeros = np.ones((8, 8), np.uint8), np.zeros((8, 8), np.uint8)
    hi = f.fuse(ones, zeros, 0.05, 0.95, 0.95, 0)
    lo = f.fuse(ones, zeros, 0.95, 0.1, 0.1, 3)
    assert hi.active == "DL" and hi.mask.all() and lo.active == "FALLBACK" and not lo.mask.any()


def test_training_stub_handles_missing_and_dummy_csv() -> None:
    cfg = smoke_config()
    assert train_fusion(cfg, csv_path=str(resolve_path("./outputs/smoke/does_not_exist.csv"))) is None
    p = resolve_path("./outputs/smoke/fusion_dummy.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["uncertainty", "geometric", "temporal", "terrain", "target_alpha"])
        for _ in range(50):
            w.writerow([*rng.random(3), int(rng.integers(0, 4)), float(rng.random() > 0.5)])
    out = train_fusion(cfg, str(p), epochs=5, out_path=str(resolve_path("./outputs/smoke/fusion_net.pt")))
    assert out is not None and resolve_path(out).exists()
