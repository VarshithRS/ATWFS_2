"""Stage 2 smoke test: forward pass, head shapes, NaN check, GroupNorm-only, CPU."""
import pytest
import torch
import torch.nn as nn

from data.dataset import build_dataloader, build_dataset
from models.unet_mobilenet import ENCODER_SPECS, build_model, evidential_uncertainty
from utils.config import smoke_config


@pytest.mark.parametrize("key", ["model", "student"])
def test_forward_all_heads(key: str) -> None:
    cfg = smoke_config()
    model = build_model(cfg[key]).eval()
    h, w = cfg["data"]["image_size"]
    ds = build_dataset(cfg, "val", synthetic=True)
    batch = next(iter(build_dataloader(ds, cfg, False, batch_size=4)))
    with torch.no_grad():
        out = model(batch["image"])
    b = 4
    assert out["seg_logits"].shape == (b, 2, h, w)
    assert out["alpha"].shape == (b, 2, h, w) and bool((out["alpha"] >= 1).all())
    assert out["terrain_logits"].shape == (b, cfg[key]["terrain_classes"])
    assert len(out["aux_logits"]) == 2
    assert out["aux_logits"][0].shape == (b, 2, h // 8, w // 8)
    assert out["aux_logits"][1].shape == (b, 2, h // 4, w // 4)
    for t in [out["seg_logits"], out["alpha"], out["terrain_logits"], *out["aux_logits"]]:
        assert torch.isfinite(t).all()
    u = evidential_uncertainty(out["alpha"])
    assert u.shape == (b, h, w) and bool((u > 0).all()) and bool((u <= 1.0 + 1e-6).all())
    assert out["seg_logits"].device.type == "cpu"


@pytest.mark.parametrize("name", list(ENCODER_SPECS))
def test_encoder_channels_match_spec_and_no_batchnorm(name: str) -> None:
    cfg = smoke_config({"model": {"encoder": name}})
    model = build_model(cfg["model"])
    feats = model.encoder(torch.zeros(1, 3, 64, 64))
    assert [f.shape[1] for f in feats] == ENCODER_SPECS[name]["channels"]
    assert [f.shape[-1] for f in feats] == [32, 16, 8, 4, 2]
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert any(isinstance(m, nn.GroupNorm) for m in model.modules())


def test_odd_input_size_still_works() -> None:
    model = build_model(smoke_config()["model"]).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 72, 100))
    assert out["seg_logits"].shape == (1, 2, 72, 100)
