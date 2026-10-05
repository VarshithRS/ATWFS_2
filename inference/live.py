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


def annotate(frame_bgr: np.ndarray, res: FrameResult, skipped: bool = False) -> np.ndarray:
    """Draw mask overlay, boxes, steering line and status text.

    Args:
        frame_bgr: Original frame.
        res: Result to visualise.
        skipped: Whether the result is re-used (frame-skip).

    Returns:
        Annotated BGR frame.
    """
    out = frame_bgr.copy()
    ov = out.copy()
    ov[res.mask > 0] = (0, 200, 0)
    out = cv2.addWeighted(ov, 0.4, out, 0.6, 0)
    h, w = out.shape[:2]
    for d in res.detections:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), (0, 0, 255), 2)
        cv2.putText(out, d.name, (d.x1, max(10, d.y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.line(out, (w // 2, h - 1), (int(res.command.centroid_x), int(0.7 * h)), (255, 255, 0), 2)
    c = res.command
    lines = [f"estimator: {res.active}{' (held)' if skipped else ''}", f"trust: {c.trust:.2f}  unc: {res.uncertainty:.2f}",
             f"angle: {c.angle:+.1f} deg  speed: {c.speed:.2f}", f"obstacle: {int(c.obstacle_flag)}  geo {res.geometric:.2f} tmp {res.temporal:.2f}"]
    for i, t in enumerate(lines):
        cv2.putText(out, t, (6, 14 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(out, t, (6, 14 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
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
                 frame_skip: int, errors: List[BaseException]) -> None:
        """Create the worker.

        Args:
            pipeline: Pipeline.
            in_q: Frame queue.
            out_q: Result queue of ``(index, annotated_frame, packet)``.
            stop: Shared stop event.
            frame_skip: Process every Nth frame; others reuse the last result.
            errors: Shared list collecting thread exceptions.
        """
        super().__init__(daemon=True, name="inference")
        self.p, self.in_q, self.out_q, self.stop = pipeline, in_q, out_q, stop
        self.skip, self.errors = max(1, int(frame_skip)), errors

    def _put(self, item: Any) -> None:
        while not self.stop.is_set():
            try:
                self.out_q.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def run(self) -> None:
        """Thread body."""
        last: Optional[FrameResult] = None
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
                self._put((idx, annotate(frame, last, skipped), last.command.packet))
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
    worker = InferenceWorker(pipeline, frame_q, result_q, stop, inf["frame_skip"], errors)
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
