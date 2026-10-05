"""Stage 6 smoke test: all compression variants produce valid output; CSV is correct."""
import csv
import math

import torch

from evaluation.benchmark import FIELDS, run_benchmark
from models.compression import prune_decoder_structured, quantize_int8_ptq
from models.unet_mobilenet import build_model, count_params
from utils.config import resolve_path, smoke_config


def test_prune_shrinks_and_still_runs() -> None:
    cfg = smoke_config()
    m = build_model(cfg["model"]).eval()
    p = prune_decoder_structured(m, 0.3)
    assert count_params(p) < count_params(m)
    out = p(torch.randn(2, 3, 64, 64))
    assert out["seg_logits"].shape == (2, 2, 64, 64) and torch.isfinite(out["seg_logits"]).all()


def test_ptq_runs() -> None:
    cfg = smoke_config()
    m = build_model(cfg["student"]).eval()
    x = torch.randn(2, 3, 64, 64)
    q, desc = quantize_int8_ptq(m, [x, x], x[:1])
    out = q(x)
    assert out["seg_logits"].shape == (2, 2, 64, 64) and torch.isfinite(out["seg_logits"]).all()
    assert desc.startswith("int8"), desc


def test_benchmark_csv() -> None:
    cfg = smoke_config({"compression": {"fps_iters": 3, "ptq_calibration_batches": 2}})
    csv_path = resolve_path(cfg["paths"]["report_dir"]) / "bench_smoke.csv"
    rows = run_benchmark(cfg, synthetic=True, out_csv=str(csv_path))
    assert [r["variant"] for r in rows] == ["original_teacher", "distilled_student", "int8_ptq", "structured_pruned"]
    with open(csv_path, newline="") as fh:
        read = list(csv.DictReader(fh))
    assert list(read[0].keys()) == FIELDS and len(read) == 4
    for r in read:
        for k in ("size_MB", "cpu_fps", "miou", "terrain_acc", "ece"):
            assert math.isfinite(float(r[k])) and float(r[k]) >= 0, (r["variant"], k, r[k])
    sizes = {r["variant"]: float(r["size_MB"]) for r in read}
    assert sizes["distilled_student"] < sizes["original_teacher"] and sizes["structured_pruned"] < sizes["original_teacher"]
    assert sizes["int8_ptq"] < sizes["original_teacher"]
