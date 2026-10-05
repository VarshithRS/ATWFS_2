"""Run every stage's smoke test in order and print a PASS/FAIL summary ("verify everything works").

    python run_all_smoke_tests.py            # all stages
    python run_all_smoke_tests.py --stop     # stop at first failure
Exit code 0 only if every stage passes. Needs no real data, GPU or network (YOLO falls back gracefully).
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from utils.logging_utils import setup_logging

logger = logging.getLogger("smoke")
ROOT = Path(__file__).resolve().parent
STAGES = [
    ("1  data pipeline", "tests/test_stage1_data.py"),
    ("2  model", "tests/test_stage2_model.py"),
    ("3  loss", "tests/test_stage3_loss.py"),
    ("4  training (2-epoch CPU fallback)", "tests/test_stage4_train.py"),
    ("5  distillation", "tests/test_stage5_distill.py"),
    ("6  compression + benchmark", "tests/test_stage6_compression.py"),
    ("7  classical fallback", "tests/test_stage7_classical.py"),
    ("8  geometric/temporal checks", "tests/test_stage8_consistency.py"),
    ("9  trust fusion (ATWFS)", "tests/test_stage9_fusion.py"),
    ("10 live video inference", "tests/test_stage10_live.py"),
    ("11 evaluation & reporting", "tests/test_stage11_eval.py"),
]


def main() -> int:
    """Run all stages.

    Returns:
        Process exit code (0 = all passed).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", action="store_true", help="stop at the first failing stage")
    a = ap.parse_args()
    setup_logging()
    results = []
    for name, test in STAGES:
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "pytest", test], cwd=ROOT, capture_output=True, text=True)
        ok = p.returncode == 0
        results.append((name, ok, time.time() - t0))
        logger.info("Stage %-36s %s (%.1fs)", name, "PASS" if ok else "FAIL", time.time() - t0)
        if not ok:
            logger.error("\n%s", (p.stdout + p.stderr)[-3000:])
            if a.stop:
                break
    n_ok = sum(r[1] for r in results)
    logger.info("=== %d/%d stages passed ===", n_ok, len(STAGES))
    return 0 if n_ok == len(STAGES) else 1


if __name__ == "__main__":
    sys.exit(main())
