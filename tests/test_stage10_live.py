"""Stage 10 smoke test: end-to-end on a synthetic clip -> valid output video + console packets."""
import re
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from data.synthetic import write_synthetic_video
from inference.decision import compute_command, subtract_obstacles
from inference.detector import Detection, NullDetector, ObstacleDetector
from inference.live import run_video
from models.unet_mobilenet import build_model
from utils.config import resolve_path, smoke_config

PACKET = re.compile(r"^\[-?\d+\.\d+, \d+\.\d+, \d\.\d+, [01]\]$")


class FakeDetector(ObstacleDetector):
    """Always reports a box in the middle of the lower image (deterministic obstacle for the test)."""

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        h, w = bgr.shape[:2]
        return [Detection(int(w * .4), int(h * .6), int(w * .6), int(h * .9), 2, "car", 0.9)]


def _clip(cfg) -> str:
    s = cfg["smoke"]
    return write_synthetic_video(str(resolve_path("./outputs/smoke/clip.mp4")), s["video_frames"], tuple(s["video_size"]),
                                 fourcc=cfg["inference"]["video_fourcc"])


def test_decision_formula_and_obstacle_subtraction() -> None:
    d = smoke_config()["inference"]["decision"]
    mask = np.zeros((100, 200), np.uint8); mask[50:, 60:140] = 1  # centred road
    free = compute_command(mask, [], trust=1.0, terrain=0, cfg=d)
    assert abs(free.angle) < 1.0 and not free.obstacle_flag and 0 < free.speed <= d["v_max"]
    assert compute_command(mask, [], 0.1, 0, d).speed == 0.0, "low trust must stop"
    assert compute_command(mask, [], 1.0, 3, d).speed < free.speed, "bad terrain must slow"
    right = np.zeros_like(mask); right[50:, 140:190] = 1
    assert compute_command(right, [], 1.0, 0, d).angle > 5, "road on the right must steer right"
    near = [Detection(80, 60, 120, 95, 2, "car", .9)]
    c = compute_command(subtract_obstacles(mask, near, 2), near, 1.0, 0, d)
    assert c.obstacle_flag and c.speed < free.speed
    assert subtract_obstacles(mask, near, 0)[70, 100] == 0


def test_end_to_end_on_synthetic_clip(capsys) -> None:
    cfg = smoke_config()
    clip = _clip(cfg)
    model = build_model(cfg["model"])
    out = str(resolve_path("./outputs/smoke/annotated.mp4"))
    res = run_video(cfg, clip, out, model=model, detector=FakeDetector())
    n = cfg["smoke"]["video_frames"]
    assert res["frames"] == n and len(res["packets"]) == n
    cap = cv2.VideoCapture(res["output_path"])
    frames = 0
    while cap.read()[0]:
        frames += 1
    cap.release()
    assert frames == n and Path(res["output_path"]).stat().st_size > 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == n and all(PACKET.match(l) for l in lines), lines[:3]
    for angle, speed, trust, flag in res["packets"]:
        assert np.isfinite([angle, speed, trust]).all() and 0 <= trust <= 1 and flag in (0, 1) and speed >= 0
    assert any(p[3] == 1 for p in res["packets"]), "fake obstacle in the corridor should raise the flag"


def test_frame_skip_and_default_detector(capsys) -> None:
    cfg = smoke_config({"inference": {"frame_skip": 3}})
    clip = _clip(cfg)
    res = run_video(cfg, clip, str(resolve_path("./outputs/smoke/annotated_skip.mp4")),
                    model=build_model(cfg["model"]), detector=NullDetector(), max_frames=7)
    assert res["frames"] == 7
    pk = res["packets"]
    assert pk[1] == pk[0] and pk[2] == pk[0], "skipped frames must hold the last packet"

def test_annotate_rendering() -> None:
    from inference.live import annotate, FrameResult
    from inference.decision import Command
    cfg = smoke_config()
    
    # Create synthetic frame and masks
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Empty mask (no corridor)
    mask_empty = np.zeros((h, w), dtype=np.uint8)
    res_empty = FrameResult(mask_empty, Command(0, 0, 0, False, 0.0, 0.0, 0.0), "TEST", 0.0, 0.0, 0.0, 0, [])
    ann1 = annotate(frame, res_empty, cfg=cfg)
    assert ann1.shape == (h, w, 3)
    
    # 2. Trapezoid mask
    mask_trap = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[200, 479], [440, 479], [380, 240], [260, 240]], dtype=np.int32)
    cv2.fillPoly(mask_trap, [pts], 1)
    res_trap = FrameResult(mask_trap, Command(0, 1.0, 1.0, False, 0.0, 0.0, 0.0), "TEST", 0.0, 0.0, 0.0, 0, [])
    ann2 = annotate(frame, res_trap, cfg=cfg)
    assert ann2.shape == (h, w, 3)
    
    # 3. Mask with car-sized hole
    mask_hole = mask_trap.copy()
    mask_hole[350:400, 300:340] = 0
    res_hole = FrameResult(mask_hole, Command(0, 1.0, 1.0, False, 0.0, 0.0, 0.0), "TEST", 0.0, 0.0, 0.0, 0, [])
    ann3 = annotate(frame, res_hole, cfg=cfg)
    assert ann3.shape == (h, w, 3)
