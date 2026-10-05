"""Stage 11 smoke test: official-protocol mIoU, failure taxonomy CSVs, Grad-CAM."""
import csv
import math
from pathlib import Path

import numpy as np
import torch

from data.dataset import build_dataloader, build_dataset
from evaluation.evaluate import official_confusion, official_ious, run_evaluation
from evaluation.failure_taxonomy import TAGS, run_taxonomy, tag_sample
from evaluation.gradcam import GradCAM, explain_images
from models.unet_mobilenet import build_model
from training.train import train
from utils.config import resolve_path, smoke_config


def test_official_miou_matches_hand_computation() -> None:
    gt = np.array([0, 0, 1, 1, 1, 255])
    pr = np.array([0, 1, 1, 1, 0, 1])
    mat = official_confusion(gt, pr, 2)
    # class0: tp=1 fn=1 fp=1 -> 1/3 ; class1: tp=2 fn=1 fp=1(row0) + 1(ignore row) = 2 -> 2/5
    ious = official_ious(mat)
    assert np.allclose(ious, [1 / 3, 2 / 5])
    k26 = official_ious(official_confusion(np.array([0, 1, 255]), np.array([0, 1, 0]), 26))
    assert np.isnan(k26[5]) and abs(k26[1] - 1.0) < 1e-9 and abs(k26[0] - 0.5) < 1e-9  # ignored px predicted road -> FP


def test_evaluation_on_synthetic_val_writes_files() -> None:
    cfg = smoke_config()
    m = run_evaluation(cfg, None, synthetic=True, model=build_model(cfg["model"]), out_dir=str(resolve_path("./outputs/smoke/eval")))
    for k in ("miou_idd", "iou_drivable", "iou_nondrivable", "terrain_acc", "ece"):
        assert math.isfinite(m[k]) and 0 <= m[k] <= 1, (k, m[k])
    rows = list(csv.reader(open(resolve_path("./outputs/smoke/eval/evaluation.csv"))))
    assert rows[0] == ["metric", "value"] and len(rows) == 6


def test_taxonomy_heuristics_and_csv() -> None:
    ev = smoke_config()["evaluation"]
    h = w = 64
    gt = np.zeros((h, w), np.uint8); gt[:, 20:44] = 1
    valid, none = np.ones_like(gt), np.zeros_like(gt)
    ok = tag_sample(gt, gt, none, valid, 1, 1, ev)
    assert ok["tags"] == [] and ok["iou_drivable"] == 1.0
    shifted = np.zeros_like(gt); shifted[:, 22:46] = 1
    assert "boundary_error" in tag_sample(shifted, gt, none, valid, 1, 1, ev)["tags"]
    assert "low_mIoU_frame" in tag_sample(1 - gt, gt, none, valid, 1, 1, ev)["tags"]
    obst = np.zeros_like(gt); obst[30:50, 25:40] = 1
    assert "missed_obstacle" in tag_sample(gt, gt, obst, valid, 1, 1, ev)["tags"]
    assert "terrain_misclassification" in tag_sample(gt, gt, none, valid, 0, 2, ev)["tags"]
    assert "terrain_misclassification" not in tag_sample(gt, gt, none, valid, 0, -1, ev)["tags"]
    cfg = smoke_config()
    loader = build_dataloader(build_dataset(cfg, "val", True), cfg, False)
    out = run_taxonomy(cfg, build_model(cfg["model"]).eval(), loader, torch.device("cpu"), str(resolve_path("./outputs/smoke/tax")), save_failures=True)
    n = cfg["smoke"]["num_val"]
    frames = list(csv.DictReader(open(out["frames_csv"])))
    assert len(frames) == n and all(r["status"] in ("ok", "wrong") for r in frames)
    summ = {r["tag"]: r for r in csv.DictReader(open(out["summary_csv"]))}
    assert set(TAGS) <= set(summ) and int(summ["total_frames"]["count"]) == n
    assert list(Path(resolve_path("./outputs/smoke/tax/failures")).glob("*.png")), "untrained model should fail somewhere"


def test_gradcam_shapes_and_range() -> None:
    cfg = smoke_config()
    model = build_model(cfg["model"]).eval()
    h, w = cfg["data"]["image_size"]
    x = torch.randn(1, 3, h, w)
    for task, layer in (("terrain", model.terrain_conv), ("seg", model.encoder.features[-1])):
        g = GradCAM(model, layer)
        cam, cls = g(x, task)
        g.close()
        assert cam.shape == (h, w) and np.isfinite(cam).all() and 0 <= cam.min() and cam.max() <= 1 + 1e-6
    out = explain_images(cfg, model, str(resolve_path("./outputs/smoke/tax/failures")), str(resolve_path("./outputs/smoke/cam")), "terrain")
    assert out and all(Path(p).exists() for p in out)
