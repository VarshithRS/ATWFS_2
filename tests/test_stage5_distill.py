"""Stage 5 smoke test: 1 epoch of distillation on synthetic data."""
import math
from pathlib import Path

import torch

from data.dataset import build_dataloader, build_dataset
from models.unet_mobilenet import build_model, count_params
from training.distill import kd_kl, make_distill_loss, train_distill
from training.train import load_model_from_checkpoint
from utils.config import smoke_config


def test_kd_kl_zero_for_identical_logits() -> None:
    x = torch.randn(2, 2, 4, 4)
    assert float(kd_kl(x, x, 2.0)) < 1e-6 and float(kd_kl(x, x + torch.randn_like(x), 2.0)) > 0


def test_distill_one_epoch_and_student_valid() -> None:
    cfg = smoke_config()
    res = train_distill(cfg, synthetic=True, epochs=1, tag="smoke_student")
    h = res["history"][0]
    assert math.isfinite(h["train_loss"]) and math.isfinite(h["val_loss"])
    assert Path(res["best_path"]).exists()
    student, ck = load_model_from_checkpoint(res["best_path"], torch.device("cpu"))
    teacher = build_model(cfg["model"])
    assert ck["model_cfg"]["encoder"] == "mobilenet_v3_small"
    assert count_params(student) < count_params(teacher)
    batch = next(iter(build_dataloader(build_dataset(cfg, "val", True), cfg, False, batch_size=3)))
    with torch.no_grad():
        out = student(batch["image"])
    hh, ww = cfg["data"]["image_size"]
    assert out["seg_logits"].shape == (3, 2, hh, ww) and out["alpha"].shape == (3, 2, hh, ww)
    assert out["terrain_logits"].shape == (3, 4) and torch.isfinite(out["seg_logits"]).all()
    # distillation term itself is finite and backpropagates into the student only
    fn = make_distill_loss(teacher.eval(), cfg["distill"])
    student.train()
    loss, info = fn(batch, student(batch["image"]))
    loss.backward()
    assert torch.isfinite(loss) and all(not p.requires_grad for p in teacher.parameters())
