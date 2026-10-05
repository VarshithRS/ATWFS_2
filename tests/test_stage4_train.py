"""Stage 4 smoke test: 2-epoch CPU-fallback training, checkpoint save + reload."""
import math
import time
from pathlib import Path

import torch

from data.dataset import build_dataloader, build_dataset
from training.train import evaluate, load_model_from_checkpoint, train
from utils.config import resolve_path, smoke_config


def test_two_epoch_cpu_fallback_and_checkpoint_reload() -> None:
    cfg = smoke_config()
    t0 = time.time()
    res = train(cfg, synthetic=True, tag="smoke", epochs=2)
    elapsed = time.time() - t0
    hist = res["history"]
    assert len(hist) == 2
    assert all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"]) for h in hist)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"] * 3, "loss exploded"
    for key in ("miou", "terrain_acc", "ece"):
        assert math.isfinite(hist[-1][key]), key
    assert Path(res["best_path"]).exists() and Path(res["last_path"]).exists()
    model, ck = load_model_from_checkpoint(res["last_path"], torch.device("cpu"))
    val = build_dataloader(build_dataset(cfg, "val", True), cfg, False)
    m = evaluate(model, val, torch.device("cpu"))
    assert abs(m["miou"] - ck["metrics"]["miou"]) < 1e-6, "reloaded model differs from saved one"
    assert (resolve_path(cfg["paths"]["report_dir"]) / "smoke_history.csv").exists()
    assert elapsed < 120, f"fallback too slow: {elapsed:.0f}s"
