"""Stage 3 smoke test: single finite scalar loss; gradients reach every parameter."""
import numpy as np
import torch

from data.dataset import build_dataloader, build_dataset
from models.unet_mobilenet import build_model
from training.losses import MultiTaskLoss, boundary_weight_map, dice_loss, evidential_loss
from utils.config import smoke_config


def _batch(cfg):
    ds = build_dataset(cfg, "train", synthetic=True, train=False)
    return next(iter(build_dataloader(ds, cfg, False, batch_size=4)))


def test_total_loss_scalar_backward_all_grads() -> None:
    cfg = smoke_config()
    model, crit = build_model(cfg["model"]), MultiTaskLoss(cfg["loss"])
    crit.set_epoch(1)
    batch = _batch(cfg)
    total, info = crit(model(batch["image"]), batch)
    assert total.dim() == 0 and torch.isfinite(total)
    assert all(np.isfinite(v) for v in info.values())
    total.backward()
    missing = [n for n, p in list(model.named_parameters()) + list(crit.named_parameters()) if p.grad is None]
    assert not missing, f"no grad for: {missing[:5]}"
    assert all(torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(crit.parameters()))


def test_all_terrain_labels_ignored_is_finite_and_has_grads() -> None:
    cfg = smoke_config()
    model, crit = build_model(cfg["model"]), MultiTaskLoss(cfg["loss"])
    batch = _batch(cfg)
    batch["terrain"] = torch.full_like(batch["terrain"], -1)
    
    # Try forcing log_vars to large negative values to check bounding
    crit.log_vars.data.fill_(-10.0)
    
    total, _ = crit(model(batch["image"]), batch)
    total.backward()
    
    assert torch.isfinite(total)
    # The loss must be bounded below because log_vars are clamped to -4
    # and terrain is skipped
    assert total.item() >= -8.0  # -4 * 2 tasks
    assert all(p.grad is not None for p in model.parameters())


def test_components() -> None:
    m = np.zeros((32, 32), np.uint8); m[:, 16:] = 1
    w = boundary_weight_map(m)
    assert w.shape == (32, 32) and w[:, 15:17].min() > w[:, 0].max() and (boundary_weight_map(np.zeros((8, 8), np.uint8)) == 1).all()
    lg, y = torch.randn(2, 2, 8, 8), torch.randint(0, 2, (2, 8, 8))
    assert 0 <= float(dice_loss(lg, y)) <= 1
    tot, fit = evidential_loss(torch.rand(2, 2, 8, 8) * 5 + 1, y, 0.5)
    assert torch.isfinite(tot) and torch.isfinite(fit)
