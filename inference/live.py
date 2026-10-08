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
import os
import shutil
import subprocess

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
        if cfg["inference"].get("use_terrain", False):
            terrain = int(out["terrain_logits"].argmax(1)[0])
        else:
            terrain = 0
        fb_mask = self.classical.estimate(small)
        geo = geometric_score(dl_mask, cfg["consistency"]["geometric"])
        tmp = temporal_score(self.prev_dl_mask, dl_mask, speed, cfg["consistency"]["temporal"])
        self.prev_dl_mask = dl_mask
        fused = self.fusion.fuse(dl_mask, fb_mask, unc, geo, tmp, terrain)
        mask = cv2.resize(fused.mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        
        hood_frac = cfg["inference"].get("hood_frac", 0.0)
        if hood_frac > 0:
            hood_h = int(fh * hood_frac)
            mask[fh - hood_h:, :] = 0
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8)

        # ---- obstacle subtraction (YOLOv8n; custom cattle detector plugs in via inference/detector.py) ----
        dets = self.detector.detect(frame_bgr)
        mask = subtract_obstacles(mask, dets, cfg["inference"]["decision"]["obstacle_dilate_px"])
        cmd = compute_command(mask, dets, fused.alpha, terrain, cfg["inference"]["decision"])
        return FrameResult(mask, cmd, fused.active, unc, geo, tmp, terrain, dets)


def annotate(frame_bgr: np.ndarray, res: FrameResult, skipped: bool = False,
             frame_idx: int = 0, total_frames: Optional[int] = None, fps: float = 0.0,
             state: Optional[Dict[str, Any]] = None, cfg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if state is None: state = {}
    if cfg is None: cfg = {}
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    
    # Scale based on frame width (base 1280)
    s = max(w / 1280.0, 0.5)
    thickness = max(1, int(round(2 * s)))
    font_scale = 0.7 * s

    # Configs
    overlay_cfg = cfg.get("inference", {}).get("overlay", {})
    bottom_frac = overlay_cfg.get("bottom_frac", 0.90)
    n_levels = overlay_cfg.get("n_levels", 8)
    merge_gap_frac = overlay_cfg.get("merge_gap_frac", 0.03)
    min_width_frac = overlay_cfg.get("min_width_frac", 0.04)
    ema_x = overlay_cfg.get("ema_x", 0.35)
    ema_y = overlay_cfg.get("ema_y", 0.20)
    show_mask = overlay_cfg.get("show_mask", False)

    gap = int(merge_gap_frac * w)
    min_width = int(min_width_frac * w)

    if show_mask:
        contours, _ = cv2.findContours(res.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 0, 255), max(1, int(s)))

    # Find drivable rows
    row_sums = res.mask.sum(axis=1)
    drivable_y = np.where(row_sums > 0)[0]
    
    corridor_found = False
    road_width_pct = 0.0

    if len(drivable_y) > 0:
        raw_y_far = drivable_y[0]
        raw_y_near = min(int(h * bottom_frac), drivable_y[-1])
        
        if "y_far" not in state:
            state["y_far"] = float(raw_y_far)
            state["y_near"] = float(raw_y_near)
            state["left_ema"] = np.zeros(n_levels)
            state["right_ema"] = np.zeros(n_levels)
            state["active_k"] = 0
        else:
            state["y_far"] = ema_y * raw_y_far + (1 - ema_y) * state["y_far"]
            state["y_near"] = ema_y * raw_y_near + (1 - ema_y) * state["y_near"]
            
        y_far = int(state["y_far"])
        y_near = int(state["y_near"])
        
        if y_near > y_far + 10:
            y_levels = np.linspace(y_near, y_far, n_levels, dtype=int)
            lefts = []
            rights = []
            prev_run = None
            
            for i, y in enumerate(y_levels):
                y = np.clip(y, 0, h - 1)
                row = res.mask[y]
                nz = np.nonzero(row)[0]
                runs = []
                if len(nz) > 0:
                    start = nz[0]
                    for j in range(1, len(nz)):
                        if nz[j] - nz[j-1] > gap:
                            runs.append([start, nz[j-1]])
                            start = nz[j]
                    runs.append([start, nz[-1]])
                
                runs = [r for r in runs if r[1] - r[0] >= min_width]
                if not runs:
                    break
                
                if prev_run is None:
                    # Bottom level: nearest to W/2
                    best_run = None
                    best_dist = float('inf')
                    for r in runs:
                        if r[0] <= w/2 <= r[1]:
                            dist = 0
                        else:
                            dist = min(abs(r[0] - w/2), abs(r[1] - w/2))
                        if dist < best_dist:
                            best_dist = dist
                            best_run = r
                    prev_run = best_run
                else:
                    # Higher levels: max overlap
                    best_run = None
                    best_overlap = 0
                    for r in runs:
                        overlap = max(0, min(r[1], prev_run[1]) - max(r[0], prev_run[0]))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_run = r
                    if best_overlap == 0:
                        break # Stop tracing
                    prev_run = best_run
                    
                lefts.append(prev_run[0])
                rights.append(prev_run[1])

            k = len(lefts)
            if k >= 2:
                corridor_found = True
                for i in range(k):
                    if i >= state["active_k"]:
                        state["left_ema"][i] = lefts[i]
                        state["right_ema"][i] = rights[i]
                    else:
                        state["left_ema"][i] = ema_x * lefts[i] + (1 - ema_x) * state["left_ema"][i]
                        state["right_ema"][i] = ema_x * rights[i] + (1 - ema_x) * state["right_ema"][i]
                state["active_k"] = k
                
                y_active = y_levels[:k]
                l_active = state["left_ema"][:k]
                r_active = state["right_ema"][:k]
                
                deg = 2 if k >= 4 else 1
                p_l = np.polyfit(y_active, l_active, deg)
                p_r = np.polyfit(y_active, r_active, deg)
                
                y_samples = np.linspace(y_active[0], y_active[-1], 24)
                l_fit = np.polyval(p_l, y_samples)
                r_fit = np.polyval(p_r, y_samples)
                
                l_fit = np.clip(l_fit, 0, w - 1)
                r_fit = np.clip(r_fit, 0, w - 1)
                r_fit = np.maximum(r_fit, l_fit + 2)
                
                road_width_pct = (r_fit[0] - l_fit[0]) / w * 100.0
                
                c_fit = (l_fit + r_fit) / 2.0
                
                poly_pts = np.concatenate([
                    np.column_stack([l_fit, y_samples]),
                    np.column_stack([r_fit, y_samples])[::-1]
                ]).astype(np.int32)
                
                overlay = out.copy()
                cv2.fillPoly(overlay, [poly_pts], (0, 200, 0), lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
                
                l_pts = np.column_stack([l_fit, y_samples]).astype(np.int32)
                r_pts = np.column_stack([r_fit, y_samples]).astype(np.int32)
                cv2.polylines(out, [l_pts], False, (0, 255, 0), thickness, cv2.LINE_AA)
                cv2.polylines(out, [r_pts], False, (0, 255, 0), thickness, cv2.LINE_AA)
                
                c_pts = np.column_stack([c_fit, y_samples]).astype(np.int32)
                cv2.polylines(out, [c_pts], False, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.polylines(out, [c_pts], False, (0, 255, 255), thickness, cv2.LINE_AA)
                
                pt1 = tuple(c_pts[-2])
                pt2 = tuple(c_pts[-1])
                cv2.arrowedLine(out, pt1, pt2, (0, 0, 0), thickness + 2, cv2.LINE_AA, tipLength=0.2)
                cv2.arrowedLine(out, pt1, pt2, (0, 255, 255), thickness, cv2.LINE_AA, tipLength=0.2)
    
    if not corridor_found:
        state.clear()
        
    for d in res.detections:
        if d.name.lower() in ["car", "truck", "bus", "vehicle", "motorcycle"]:
            # Basic tracking to see if it's growing/approaching
            if "prev_dets" in state:
                for pd in state["prev_dets"]:
                    iou = max(0, min(d.x2, pd.x2) - max(d.x1, pd.x1)) * max(0, min(d.y2, pd.y2) - max(d.y1, pd.y1)) / ((d.x2 - d.x1)*(d.y2 - d.y1) + (pd.x2 - pd.x1)*(pd.y2 - pd.y1) + 1e-6)
                    if iou > 0.3 and (d.y2 > pd.y2 + 2 or (d.x2 - d.x1) > (pd.x2 - pd.x1) * 1.02):
                        d.name = "APPROACHING " + d.name
                        break
        state["prev_dets"] = res.detections
        
        color = (0, 165, 255) if "APPROACHING" in d.name else (0, 0, 255)
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, thickness)
        label = f"{d.name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lbl_y = max(th + 5, d.y1 - 5)
        cv2.putText(out, label, (d.x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(out, label, (d.x1, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    frames_str = f"{frame_idx}/{total_frames}" if total_frames else f"{frame_idx}"
    ang = res.command.angle
    curve = "STRAIGHT" if abs(ang) < 5 else ("RIGHT SHARP" if ang > 15 else ("RIGHT" if ang > 5 else ("LEFT SHARP" if ang < -15 else "LEFT")))
    
    lines = [
        f"ATWFS DRIVABLE REGION | FRAME: {frames_str} | {fps:.1f} FPS",
        f"ESTIMATOR: {res.active}   TRUST: {res.command.trust:.2f}",
        f"STEER: {ang:+.1f} deg {curve}   SPEED: {res.command.speed:.2f} m/s",
        f"OBSTACLE: {'YES' if res.command.obstacle_flag else 'NO'}",
        f"ROAD WIDTH: {road_width_pct:.1f}%",
        "LEGEND: Green=Corridor, Yellow=Path"
    ]
    
    y = int(25 * s)
    for line in lines:
        cv2.putText(out, line, (int(15*s), y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, line, (int(15*s), y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += int(30 * s)

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
                    state=self.annot_state,
                    cfg=self.p.cfg
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
                try:
                    cv2.imshow("drivable region", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop.set()
                except Exception:
                    logger.warning("cv2.imshow failed. install opencv-python for --display")
                    display = False
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("Interrupted by user")
        stop.set()
    finally:
        stop.set()
        grabber.join(timeout=5); worker.join(timeout=5)
        if writer is not None:
            writer.release()
        if display:  # pragma: no cover
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    if errors:
        raise RuntimeError(f"Inference thread crashed: {errors[0]!r}") from errors[0]
        
    # FFmpeg re-encoding
    if out_path.exists() and writer is not None:
        import shutil
        import subprocess
        import os
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = None
                
        if ffmpeg_exe:
            temp_out = out_path.with_suffix(".tmp.mp4")
            cmd = [
                ffmpeg_exe, "-y", "-i", str(out_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-preset", "veryfast",
                "-movflags", "+faststart", str(temp_out)
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(str(temp_out), str(out_path))
                logger.info("Successfully re-encoded to H.264")
            except Exception as e:
                logger.warning(f"FFmpeg re-encoding failed: {e}. Kept original file.")
                if temp_out.exists():
                    temp_out.unlink()
        else:
            logger.warning("FFmpeg not found. Kept original file. Install ffmpeg or imageio-ffmpeg for H.264 encoding.")

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
