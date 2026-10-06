"""Live / offline video inference with a threaded producer-consumer pipeline.

    grab thread --frame_q--> inference thread --result_q--> main thread (video writer / optional display / packets)

Per processed frame:
  model -> (DL mask, evidential uncertainty, terrain) ; classical fallback ; geometric + temporal checks ;
  ATWFS trust fusion ; YOLO obstacle subtraction ; steering angle + speed ; annotated frame + packet.

The ``[angle, speed, trust_score, obstacle_flag]`` packet is printed per output frame. THAT PRINT IS A STUB for a
future Arduino serial link - NO serial / hardware / pyserial code exists in this project.

Usage:
    python -m inference.live --source path/to/video.mp4 --output outputs/inference/out.mp4 --checkpoint <ckpt.pt>
    python -m inference.live --source 0                      # webcam (also rtsp:// or http:// IP-camera URLs)
"""
from __future__ import annotations

import argparse
import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from inference.classical import ClassicalFreeSpaceEstimator
from inference.consistency import geometric_score, temporal_score
from inference.decision import Command, compute_command, subtract_obstacles
from inference.detector import Detection, ObstacleDetector, build_detector
from inference.fusion import ATWFS
from models.unet_mobilenet import build_model, evidential_uncertainty
from training.train import load_model_from_checkpoint
from utils.config import load_config, resolve_path
from utils.logging_utils import setup_logging
from utils.seed import get_device, set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Everything computed for one processed frame."""

    mask: np.ndarray                    # final drivable mask at frame resolution (uint8)
    command: Command
    active: str                         # "DL" or "FALLBACK"
    uncertainty: float
    geometric: float
    temporal: float
    terrain: int
    detections: List[Detection] = field(default_factory=list)


class LivePipeline:
    """Stateful per-frame processor (keeps the previous DL mask for the temporal check)."""

    def __init__(self, cfg: Dict[str, Any], model: torch.nn.Module, device: torch.device,
                 detector: ObstacleDetector, fusion: Optional[ATWFS] = None) -> None:
        """Create the pipeline.

        Args:
            cfg: Full config.
            model: Trained multi-task model.
            device: Device.
            detector: Obstacle detector.
            fusion: Optional ATWFS (defaults to the heuristic-initialised placeholder).
        """
        self.cfg, self.model, self.device, self.detector = cfg, model.to(device).eval(), device, detector
        self.size = tuple(cfg["data"]["image_size"])  # (H, W)
        self.mean = np.asarray(cfg["data"]["imagenet_mean"], np.float32)
        self.std = np.asarray(cfg["data"]["imagenet_std"], np.float32)
        self.classical = ClassicalFreeSpaceEstimator(cfg["classical"])
        self.fusion = fusion or ATWFS(cfg["fusion"])
        self.prev_dl_mask: Optional[np.ndarray] = None

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray, speed_mps: Optional[float] = None) -> FrameResult:
        """Run the full pipeline on one BGR frame.

        Args:
            frame_bgr: Input frame.
            speed_mps: Vehicle speed for the temporal check (default ``inference.speed_mps_input``;
                STUB: replace by real odometry later).

        Returns:
            :class:`FrameResult`.
        """
        cfg = self.cfg
        speed = cfg["inference"]["speed_mps_input"] if speed_mps is None else speed_mps
        fh, fw = frame_bgr.shape[:2]
        mh, mw = self.size
        small = cv2.resize(frame_bgr, (mw, mh), interpolation=cv2.INTER_AREA)
        x = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.mean) / self.std
        out = self.model(torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device))
        dl_mask = out["seg_logits"].argmax(1)[0].cpu().numpy().astype(np.uint8)
        unc = float(evidential_uncertainty(out["alpha"]).mean())
        terrain = int(out["terrain_logits"].argmax(1)[0])
        fb_mask = self.classical.estimate(small)
        geo = geometric_score(dl_mask, cfg["consistency"]["geometric"])
        tmp = temporal_score(self.prev_dl_mask, dl_mask, speed, cfg["consistency"]["temporal"])
        self.prev_dl_mask = dl_mask
        fused = self.fusion.fuse(dl_mask, fb_mask, unc, geo, tmp, terrain)
        mask = cv2.resize(fused.mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        # ---- obstacle subtraction (YOLOv8n; custom cattle detector plugs in via inference/detector.py) ----
        dets = self.detector.detect(frame_bgr)
        mask = subtract_obstacles(mask, dets, cfg["inference"]["decision"]["obstacle_dilate_px"])
        cmd = compute_command(mask, dets, fused.alpha, terrain, cfg["inference"]["decision"])
        return FrameResult(mask, cmd, fused.active, unc, geo, tmp, terrain, dets)


def annotate(frame_bgr: np.ndarray, res: FrameResult, skipped: bool = False,
             frame_idx: int = 0, total_frames: Optional[int] = None, fps: float = 0.0,
             state: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Draw mask overlay, boxes, steering line and status text matching the sample output."""
    if state is None:
        state = {}

    out = frame_bgr.copy()
    ov = out.copy()
    h, w = out.shape[:2]

    # Highly stable mask using EMA
    curr_mask = (res.mask > 0).astype(np.float32)
    if "ema_mask" not in state:
        state["ema_mask"] = curr_mask
    else:
        state["ema_mask"] = 0.6 * curr_mask + 0.4 * state["ema_mask"]
    
    stable_mask = (state["ema_mask"] > 0.5).astype(np.uint8) * 255

    # Fill drivable area with green (with alpha blending)
    ov[stable_mask > 0] = (0, 200, 0)
    out = cv2.addWeighted(ov, 0.4, out, 0.6, 0)

    # Incoming vehicle tracking
    if "prev_dets" in state:
        prev_dets = state["prev_dets"]
        for d in res.detections:
            if d.name.lower() in ["car", "truck", "bus", "vehicle", "motorcycle"]:
                best_iou = 0.0
                best_pd = None
                for pd in prev_dets:
                    ix1, iy1 = max(d.x1, pd.x1), max(d.y1, pd.y1)
                    ix2, iy2 = min(d.x2, pd.x2), min(d.y2, pd.y2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    if iw > 0 and ih > 0:
                        inter = iw * ih
                        uni = (d.x2 - d.x1)*(d.y2 - d.y1) + (pd.x2 - pd.x1)*(pd.y2 - pd.y1) - inter
                        iou = inter / uni
                        if iou > best_iou:
                            best_iou = iou
                            best_pd = pd
                if best_iou > 0.3 and best_pd is not None:
                    # Vehicle is moving down (closer) or getting wider
                    if d.y2 > best_pd.y2 + 2 or (d.x2 - d.x1) > (best_pd.x2 - best_pd.x1) * 1.02:
                        d.name = "INCOMING " + d.name
    state["prev_dets"] = res.detections

    # Draw bounding boxes
    for d in res.detections:
        color = (0, 165, 255) if "INCOMING" in d.name else (255, 0, 0) # Orange for incoming, Blue otherwise
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        label = f"{d.name.upper()} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw box explicitly ABOVE the car
        lbl_y2 = d.y1
        lbl_y1 = lbl_y2 - th - 10
        if lbl_y1 < 0: # push it inside the box if it goes out of frame
            lbl_y1 = d.y1
            lbl_y2 = d.y1 + th + 10
            
        cv2.rectangle(out, (d.x1, lbl_y1), (d.x1 + tw + 4, lbl_y2), (0, 0, 0), -1)
        cv2.putText(out, label, (d.x1 + 2, lbl_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Determine curve type
    angle = res.command.angle
    if abs(angle) < 5:
        curve_str = "STRAIGHT"
    elif angle > 15:
        curve_str = "RIGHT SHARP"
    elif angle > 5:
        curve_str = "RIGHT"
    elif angle < -15:
        curve_str = "LEFT SHARP"
    else:
        curve_str = "LEFT"

    # ---------------------------------------------------------
    # "CRAZY STABLE" STRUCTURAL TRACKING
    # ---------------------------------------------------------
    row_sums = res.mask.sum(axis=1)
    drivable_y = np.where(row_sums > 0)[0]
    
    near_w = mid_w = far_w = 0
    avg_w = 0.0

    if len(drivable_y) > 0:
        raw_y_far = drivable_y[0]
        raw_y_near = drivable_y[-1]
        
        # 1. Extremely heavy smoothing on the horizon/bottom levels
        alpha_y = 0.05
        if "y_far" not in state:
            state["y_far"] = float(raw_y_far)
            state["y_near"] = float(raw_y_near)
        else:
            state["y_far"] = alpha_y * raw_y_far + (1 - alpha_y) * state["y_far"]
            state["y_near"] = alpha_y * raw_y_near + (1 - alpha_y) * state["y_near"]
            
        y_far = int(state["y_far"])
        y_near = int(state["y_near"])
        
        # 2. Define 4 fixed structural Y-levels (produces 3 straight segments per side)
        y_levels = np.linspace(y_near, y_far, 4, dtype=int)
        
        raw_pts = []
        widths_raw = []
        for y in y_levels:
            # find closest valid row in the raw mask
            closest_y = drivable_y[np.argmin(np.abs(drivable_y - y))]
            nz = np.nonzero(res.mask[closest_y])[0]
            if len(nz) > 0:
                lx, rx = nz[0], nz[-1]
                cx = (lx + rx) / 2.0
                widths_raw.append(rx - lx)
            else:
                lx, rx, cx = w // 2 - 50, w // 2 + 50, w // 2
            raw_pts.extend([lx, rx, cx])
            
        if widths_raw:
            avg_w = np.mean(widths_raw)
            
        raw_pts = np.array(raw_pts, dtype=np.float32)
        
        # 3. Apply heavy EMA to the X coordinates
        alpha_x = 0.15 # Low alpha for rock-solid stability
        if "track_pts" not in state:
            state["track_pts"] = raw_pts
        else:
            state["track_pts"] = alpha_x * raw_pts + (1 - alpha_x) * state["track_pts"]
            
        pts = state["track_pts"].astype(np.int32)
        
        # pts has 12 values: [l0, r0, c0, l1, r1, c1, l2, r2, c2, l3, r3, c3]
        l0, r0, c0 = pts[0], pts[1], pts[2]
        l1, r1, c1 = pts[3], pts[4], pts[5]
        l2, r2, c2 = pts[6], pts[7], pts[8]
        l3, r3, c3 = pts[9], pts[10], pts[11]
        
        y0, y1, y2, y3 = y_levels
        
        # For text boxes
        near_w = r0 - l0
        mid_w = r1 - l1
        far_w = r3 - l3
        
        # 4. Draw Magenta Box (Left -> Top Horizon -> Right)
        box_pts = np.array([
            [l0, y0], [l1, y1], [l2, y2], [l3, y3], # Left boundary
            [r3, y3],                               # Top horizontal connection
            [r2, y2], [r1, y1], [r0, y0]            # Right boundary
        ], dtype=np.int32)
        cv2.polylines(out, [box_pts], isClosed=False, color=(255, 0, 255), thickness=3)
        
        # 5. Draw Yellow Center Line (Ego -> Near -> Mid -> Far)
        center_pts = np.array([
            [w // 2, h - 1], # Start at ego vehicle center
            [c0, y0], [c1, y1], [c2, y2], [c3, y3]
        ], dtype=np.int32)
        cv2.polylines(out, [center_pts], isClosed=False, color=(0, 255, 255), thickness=3)

    # Draw Text Boxes
    def draw_box(img, lines, x, y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thick = 2
        pad = 8
        max_w = 0
        total_h = pad
        for text in lines:
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            max_w = max(max_w, tw)
            total_h += th + pad
        
        cv2.rectangle(img, (x, y), (x + max_w + 2 * pad, y + total_h), (0, 0, 0), -1)
        
        cur_y = y + pad
        for text in lines:
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.putText(img, text, (x + pad, cur_y + th), font, scale, (255, 255, 255), thick)
            cur_y += th + pad
        
        return y + total_h + pad

    cur_box_y = 20
    
    # Box 1
    frames_str = f"{frame_idx}/{total_frames}" if total_frames else f"{frame_idx}"
    box1_lines = [
        "YOLOPv2 + LANE-LESS REASONING",
        f"FRAME: {frames_str}",
        f"PROCESSING: {fps:.2f} FPS"
    ]
    cur_box_y = draw_box(out, box1_lines, 20, cur_box_y)
    
    # Box 2
    box2_lines = [
        "MODE: LANE_BASED",
        f"CURVE: {curve_str}"
    ]
    cur_box_y = draw_box(out, box2_lines, 20, cur_box_y)

    # Box 3
    box3_lines = [
        f"WIDTH: near={near_w}, mid={mid_w}, far={far_w}, avg={avg_w:.1f} px"
    ]
    draw_box(out, box3_lines, 20, cur_box_y)

    return out


class FrameGrabber(threading.Thread):
    """Producer thread: reads frames from a ``cv2.VideoCapture`` into a bounded queue.

    Files: blocking put (no frame is dropped). Live streams (webcam / RTSP / HTTP): when the queue is full the
    OLDEST frame is dropped so the pipeline always sees the freshest frame.
    """

    def __init__(self, source: Union[int, str], out_q: "queue.Queue", stop: threading.Event, live: bool,
                 max_frames: Optional[int] = None) -> None:
        """Open the capture (in the calling thread, so errors surface immediately).

        Args:
            source: File path, camera index or stream URL.
            out_q: Destination queue of ``(index, frame)``; ``None`` marks the end.
            stop: Shared stop event.
            live: True for webcam/IP streams.
            max_frames: Optional frame limit.
        """
        super().__init__(daemon=True, name="frame-grabber")
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")
        self.out_q, self.stop, self.live, self.max_frames = out_q, stop, live, max_frames
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not live else None

    def run(self) -> None:
        """Thread body."""
        i = 0
        try:
            while not self.stop.is_set() and (self.max_frames is None or i < self.max_frames):
                ok, frame = self.cap.read()
                if not ok:
                    break
                while not self.stop.is_set():
                    try:
                        self.out_q.put((i, frame), timeout=0.05)
                        break
                    except queue.Full:
                        if self.live:
                            try:
                                self.out_q.get_nowait()
                            except queue.Empty:
                                pass
                i += 1
        finally:
            self.cap.release()
            while True:
                try:
                    self.out_q.put(None, timeout=0.05)
                    break
                except queue.Full:
                    if self.stop.is_set():
                        try:
                            self.out_q.get_nowait()
                        except queue.Empty:
                            pass


class InferenceWorker(threading.Thread):
    """Consumer thread: runs the pipeline (every ``frame_skip``-th frame) and emits annotated frames."""

    def __init__(self, pipeline: LivePipeline, in_q: "queue.Queue", out_q: "queue.Queue", stop: threading.Event,
                 frame_skip: int, errors: List[BaseException], total_frames: Optional[int] = None) -> None:
        """Create the worker."""
        super().__init__(daemon=True, name="inference")
        self.p, self.in_q, self.out_q, self.stop = pipeline, in_q, out_q, stop
        self.skip, self.errors = max(1, int(frame_skip)), errors
        self.total_frames = total_frames
        self.annot_state: Dict[str, Any] = {}

    def _put(self, item: Any) -> None:
        while not self.stop.is_set():
            try:
                self.out_q.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def run(self) -> None:
        """Thread body."""
        import time
        last: Optional[FrameResult] = None
        start_time = time.time()
        frames_processed = 0
        try:
            while not self.stop.is_set():
                try:
                    item = self.in_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    break
                idx, frame = item
                skipped = last is not None and idx % self.skip != 0
                if not skipped:
                    last = self.p.process_frame(frame)
                
                frames_processed += 1
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed if elapsed > 0 else 0.0

                annotated = annotate(
                    frame, last, skipped,
                    frame_idx=idx + 1,
                    total_frames=self.total_frames,
                    fps=fps,
                    state=self.annot_state
                )
                self._put((idx, annotated, last.command.packet))
        except BaseException as e:  # noqa: BLE001 - propagate to main thread
            logger.exception("Inference thread failed")
            self.errors.append(e)
            self.stop.set()
        finally:
            while True:
                try:
                    self.out_q.put(None, timeout=0.05)
                    break
                except queue.Full:
                    try:
                        self.out_q.get_nowait()
                    except queue.Empty:
                        pass


def _open_writer(path: Path, fps: float, size_wh: Tuple[int, int], fourcc: str) -> Tuple[cv2.VideoWriter, Path]:
    """Open a video writer, falling back to MJPG/.avi if the preferred codec is unavailable.

    Args:
        path: Desired output path.
        fps: Frame rate.
        size_wh: ``(W, H)``.
        fourcc: Preferred FOURCC.

    Returns:
        ``(writer, actual_path)``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size_wh)
    if wr.isOpened():
        return wr, path
    alt = path.with_suffix(".avi")
    logger.warning("Codec '%s' unavailable, falling back to MJPG %s", fourcc, alt)
    wr = cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*"MJPG"), fps, size_wh)
    if not wr.isOpened():
        raise RuntimeError("Could not open any video writer")
    return wr, alt


def run_video(cfg: Dict[str, Any], source: Union[int, str], output_path: str, model: Optional[torch.nn.Module] = None,
              checkpoint: Optional[str] = None, detector: Optional[ObstacleDetector] = None,
              max_frames: Optional[int] = None, display: Optional[bool] = None) -> Dict[str, Any]:
    """Run the live pipeline on a video file or camera stream.

    Args:
        cfg: Full config.
        source: Video path, camera index or stream URL.
        output_path: Annotated video output path.
        model: Optional ready model (otherwise loaded from ``checkpoint`` / ``inference.checkpoint``).
        checkpoint: Checkpoint path.
        detector: Optional detector override.
        max_frames: Optional frame limit.
        display: Show a live window (default ``inference.display``); needs a GUI-enabled OpenCV.

    Returns:
        ``{"output_path", "frames", "packets": [(angle, speed, trust, flag), ...]}``.
    """
    set_global_seed(cfg["seed"])
    device = get_device()
    inf = cfg["inference"]
    if model is None:
        ck = Path(checkpoint) if checkpoint else resolve_path(inf["checkpoint"])
        if ck.exists():
            model, _ = load_model_from_checkpoint(str(ck), device)
        else:
            logger.warning("Checkpoint %s not found -> running with an UNTRAINED model (pipeline demo only).", ck)
            model = build_model({**cfg["model"], "pretrained": False})
    detector = detector or build_detector(inf["detector"], cfg["paths"]["output_dir"])
    pipeline = LivePipeline(cfg, model, device, detector)
    live = isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))
    stop, errors = threading.Event(), []  # type: threading.Event, List[BaseException]
    frame_q, result_q = queue.Queue(inf["queue_size"]), queue.Queue(inf["queue_size"])
    grabber = FrameGrabber(source, frame_q, stop, live, max_frames)
    worker = InferenceWorker(pipeline, frame_q, result_q, stop, inf["frame_skip"], errors, grabber.total_frames)
    grabber.start(); worker.start()
    display = inf["display"] if display is None else display
    writer, out_path, packets = None, Path(output_path), []
    try:
        while True:
            try:
                item = result_q.get(timeout=0.1)
            except queue.Empty:
                if stop.is_set() and not worker.is_alive():
                    break
                continue
            if item is None:
                break
            idx, frame, pkt = item
            if writer is None:
                writer, out_path = _open_writer(out_path, grabber.fps, (frame.shape[1], frame.shape[0]), inf["video_fourcc"])
            writer.write(frame)
            packets.append(pkt)
            # STUB for a future Arduino serial link: just a clean console line, no hardware code.
            print(f"[{pkt[0]:.2f}, {pkt[1]:.3f}, {pkt[2]:.3f}, {pkt[3]}]", flush=True)
            if display:  # pragma: no cover - needs a GUI
                cv2.imshow("drivable region", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop.set()
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("Interrupted by user")
        stop.set()
    finally:
        stop.set()
        grabber.join(timeout=5); worker.join(timeout=5)
        if writer is not None:
            writer.release()
        if display:  # pragma: no cover
            cv2.destroyAllWindows()
    if errors:
        raise RuntimeError(f"Inference thread crashed: {errors[0]!r}") from errors[0]
    logger.info("Processed %d frames -> %s", len(packets), out_path)
    return {"output_path": str(out_path), "frames": len(packets), "packets": packets}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Live drivable-region inference (video file / webcam / IP camera).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True, help="video path, camera index (0), or rtsp/http URL")
    ap.add_argument("--output", default="./outputs/inference/annotated.mp4")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--frame-skip", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--display", action="store_true")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    if a.frame_skip:
        cfg["inference"]["frame_skip"] = a.frame_skip
    src: Union[int, str] = int(a.source) if a.source.isdigit() else a.source
    run_video(cfg, src, str(resolve_path(a.output)), checkpoint=a.checkpoint, max_frames=a.max_frames, display=a.display or None)


if __name__ == "__main__":
    main()
