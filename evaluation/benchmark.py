"""Compression comparison benchmark.

Variants: original (teacher) | distilled student (Stage 5) | INT8 PTQ | structured pruning.
Reports model size (MB), CPU FPS (batch 1), mIoU / terrain accuracy / ECE on the eval split.
Output: CSV (``paths.report_dir/compression_benchmark.csv``) + log-printed table.

Note: INT8-PTQ and pruning are applied to the *original* (teacher) model by default (``--base student``
to compress the distilled student instead).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from data.dataset import build_dataloader, build_dataset
from models.compression import cpu_fps, model_size_mb, prune_decoder_structured, quantize_int8_ptq
from models.unet_mobilenet import build_model, count_params
from training.distill import load_teacher, make_distill_loss
from training.losses import MultiTaskLoss
from training.train import evaluate, load_model_from_checkpoint, run_training
from utils.config import load_config, resolve_path, smoke_config
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)
FIELDS = ["variant", "method", "params_M", "size_MB", "cpu_fps", "miou", "iou_drivable", "pixel_acc", "terrain_acc", "ece"]


def _load_student(cfg: Dict[str, Any], device: torch.device, ckpt: Optional[str]) -> torch.nn.Module:
    """Load the distilled student (random init + warning if no checkpoint exists).

    Args:
        cfg: Full config.
        device: Device.
        ckpt: Optional checkpoint path.

    Returns:
        Student model (eval).
    """
    p = Path(ckpt) if ckpt else resolve_path(cfg["paths"]["checkpoint_dir"]) / "student_best.pt"
    if p.exists():
        return load_model_from_checkpoint(str(p), device)[0]
    logger.warning("Student checkpoint %s not found -> UNTRAINED student (pipeline test only).", p)
    return build_model({**cfg["student"], "pretrained": False}).to(device).eval()


def run_benchmark(cfg: Dict[str, Any], synthetic: bool = False, teacher_ckpt: Optional[str] = None,
                  student_ckpt: Optional[str] = None, base: str = "teacher", out_csv: Optional[str] = None
                  ) -> List[Dict[str, Any]]:
    """Run the full comparison.

    Args:
        cfg: Full config.
        synthetic: Use synthetic data.
        teacher_ckpt: Teacher checkpoint.
        student_ckpt: Student checkpoint.
        base: Which model INT8/pruning start from (``teacher`` or ``student``).
        out_csv: CSV path (default under ``paths.report_dir``).

    Returns:
        List of result rows (one per variant).
    """
    set_global_seed(cfg["seed"])
    device = get_device()
    c = cfg["compression"]
    if c["cpu_threads"]:
        torch.set_num_threads(int(c["cpu_threads"]))
    split = "val" if synthetic else cfg["data"]["eval_split"]
    val_loader = build_dataloader(build_dataset(cfg, split, synthetic), cfg, False)
    teacher, student = load_teacher(cfg, device, teacher_ckpt), _load_student(cfg, device, student_ckpt)
    src = teacher if base == "teacher" else student
    h, w = cfg["data"]["image_size"]
    calib = [b["image"] for b in itertools.islice(val_loader, c["ptq_calibration_batches"])]
    q_model, q_desc = quantize_int8_ptq(src, calib, calib[0][:1])
    pruned = prune_decoder_structured(student, c["prune_amount"])
    if c["prune_finetune_epochs"] > 0:
        tr = build_dataloader(build_dataset(cfg, "train", synthetic), cfg, True)
        old_lr = cfg["train"]["lr"]
        cfg["train"]["lr"] = old_lr / 10.0
        distill_loss = make_distill_loss(teacher.to(device), cfg["distill"])
        run_training(cfg, pruned, MultiTaskLoss(cfg["loss"]), tr, val_loader, device, "pruned", cfg["student"], c["prune_finetune_epochs"], distill_loss)
        cfg["train"]["lr"] = old_lr
        pruned = pruned.cpu().eval()
    variants = [("original_teacher", "fp32", teacher, device), ("distilled_student", "fp32", student, device),
                ("int8_ptq", q_desc, q_model, torch.device("cpu")),
                ("structured_pruned", f"decoder_L1_filter_prune_{c['prune_amount']:.2f}", pruned, device)]
    rows: List[Dict[str, Any]] = []
    for name, method, model, dev in variants:
        m = evaluate(model.to(dev) if "int8" not in name else model, val_loader, dev)
        row = {"variant": name, "method": method, "params_M": round(count_params(model) / 1e6, 3) if "int8" not in name else float("nan"),
               "size_MB": round(model_size_mb(model), 3), "cpu_fps": round(cpu_fps(model, (1, 3, h, w), c["fps_warmup"], c["fps_iters"]), 2),
               **{k: round(m[k], 4) for k in ("miou", "iou_drivable", "pixel_acc", "terrain_acc", "ece")}}
        rows.append(row)
        logger.info("benchmarked %s", name)
    out = Path(out_csv) if out_csv else resolve_path(cfg["paths"]["report_dir"]) / "compression_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS); wr.writeheader(); wr.writerows(rows)
    logger.info("\n%s", format_table(rows))
    logger.info("CSV written to %s", out)
    return rows


def format_table(rows: List[Dict[str, Any]]) -> str:
    """Render rows as a fixed-width text table.

    Args:
        rows: Result rows.

    Returns:
        Table string.
    """
    widths = {f: max(len(f), *(len(str(r[f])) for r in rows)) for f in FIELDS}
    line = " | ".join(f.ljust(widths[f]) for f in FIELDS)
    return "\n".join([line, "-+-".join("-" * widths[f] for f in FIELDS)] +
                     [" | ".join(str(r[f]).ljust(widths[f]) for f in FIELDS) for r in rows])


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Compression comparison benchmark.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--student", default=None)
    ap.add_argument("--base", choices=["teacher", "student"], default="teacher")
    a = ap.parse_args()
    setup_logging()
    run_benchmark(load_config(a.config), a.synthetic, a.teacher, a.student, a.base)


if __name__ == "__main__":
    main()
