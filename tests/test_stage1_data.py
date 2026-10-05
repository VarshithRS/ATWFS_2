"""Stage 1 smoke test: synthetic data, label collapsing, augmentation."""
import numpy as np
import pytest

from data.augment import StandardAugment, SyntheticConditionAugment
from data.dataset import build_dataset
from data.labels import collapse_to_binary
from utils.config import smoke_config


def test_collapse_labels() -> None:
    m = np.array([[0, 1, 2, 3], [4, 9, 255, 25]], np.uint8)
    out = collapse_to_binary(m, [0, 1])
    assert out.tolist() == [[1, 1, 0, 0], [0, 0, 0, 0]]


def test_synthetic_samples_shapes_and_binary_labels() -> None:
    cfg = smoke_config()
    h, w = cfg["data"]["image_size"]
    ds = build_dataset(cfg, "train", synthetic=True, train=False)
    for i in range(5):
        s = ds[i]
        assert tuple(s["image"].shape) == (3, h, w)
        assert tuple(s["mask"].shape) == (h, w)
        assert set(np.unique(s["mask"].numpy()).tolist()) <= {0, 1}
        assert set(np.unique(s["obstacle"].numpy()).tolist()) <= {0, 1}
        assert 0 <= int(s["terrain"]) <= 3
    assert any(ds[i]["mask"].sum() > 0 for i in range(5)), "no drivable pixels at all"


def test_augmentation_keeps_shapes_all_effects() -> None:
    cfg = smoke_config()
    aug_cfg = cfg["augmentation"]
    ds = build_dataset(cfg, "val", synthetic=True, train=False)
    img, raw, _ = ds.source.load(0)
    std = StandardAugment({**aug_cfg["standard"], "hflip_p": 1.0, "crop_p": 1.0, "jitter_p": 1.0}, seed=1)
    i2, (m2,) = std(img, [raw])
    assert i2.shape == img.shape and m2.shape == raw.shape and i2.dtype == np.uint8
    syn = SyntheticConditionAugment(aug_cfg["synthetic"], seed=1)
    for fn, arg in [(syn.fog, 0.6), (syn.motion_blur, 9), (syn.water_reflection, 0.5), (syn.dust, 0.4)]:
        out = fn(img, arg)
        assert out.shape == img.shape and out.dtype == np.uint8
    all_on = {**aug_cfg["synthetic"], "enabled": True}
    for k in ("fog", "motion_blur", "water_reflection", "dust"):
        all_on[k] = {**all_on[k], "p": 1.0}
    assert SyntheticConditionAugment(all_on, 2)(img).shape == img.shape
    off = SyntheticConditionAugment({**aug_cfg["synthetic"], "enabled": False}, 3)
    assert np.array_equal(off(img), img), "disabled module must be a no-op"


def test_train_dataset_with_all_augmentation_enabled() -> None:
    cfg = smoke_config({"augmentation": {"synthetic": {"enabled": True}}})
    ds = build_dataset(cfg, "train", synthetic=True)
    h, w = cfg["data"]["image_size"]
    for i in range(5):
        s = ds[i]
        assert tuple(s["image"].shape) == (3, h, w) and tuple(s["mask"].shape) == (h, w)
        assert set(np.unique(s["mask"].numpy()).tolist()) <= {0, 1}
