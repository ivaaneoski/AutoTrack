"""Batch video processing, video export, and telemetry indexing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
import cv2
import numpy as np

from pathdemo.config import AppConfig
from pathdemo.pipeline import PathFollowingPipeline
from pathdemo.telemetry import TelemetryLogger


@dataclass
class VideoBatchExportResult:
    """Summary and artifacts generated from an end-to-end video batch run."""
    input_path: Path
    annotated_video_path: Path
    mask_video_path: Path
    side_by_side_video_path: Path
    csv_log_path: Path
    total_frames: int
    fps: float
    duration_sec: float
    records: List[Dict[str, Any]]


def process_and_export_video(
    video_path: str | Path,
    output_dir: str | Path = "results",
    config: Optional[AppConfig] = None,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> VideoBatchExportResult:
    """Processes an entire video file, writing annotated and mask video files,

    along with full CSV telemetry logs and indexed frame metadata.
    """
    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video stream dimensions or frame count for: {input_path}")

    out_base = Path(output_dir)
    video_out_dir = out_base / "exported_videos"
    log_out_dir = out_base / "logs"
    video_out_dir.mkdir(parents=True, exist_ok=True)
    log_out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    annotated_path = video_out_dir / f"{stem}_annotated.mp4"
    mask_path = video_out_dir / f"{stem}_mask.mp4"
    side_by_side_path = video_out_dir / f"{stem}_side_by_side.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    annotated_writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
    mask_writer = cv2.VideoWriter(str(mask_path), fourcc, fps, (width, height))
    
    # Side-by-side composite: [Annotated (width) | Mask (width)] -> (width * 2, height)
    side_by_side_writer = cv2.VideoWriter(str(side_by_side_path), fourcc, fps, (width * 2, height))

    if config is None:
        config = AppConfig()

    logger = TelemetryLogger(log_dir=str(log_out_dir), prefix=f"export_{stem}")
    pipeline = PathFollowingPipeline(config, telemetry_logger=logger)

    records: List[Dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            res = pipeline.process_frame(frame, source_name=input_path.name)
            frame_idx += 1

            # 1. Write Annotated Video
            annotated_writer.write(res.annotated_frame)

            # 2. Write Mask Video (convert 1-channel binary to 3-channel BGR)
            mask_bgr = cv2.cvtColor(res.mask, cv2.COLOR_GRAY2BGR)
            mask_writer.write(mask_bgr)

            # 3. Write Side-by-side composite
            composite = np.hstack([res.annotated_frame, mask_bgr])
            side_by_side_writer.write(composite)

            # Store record for fast interactive indexing
            rec = {
                "frame": frame_idx,
                "timestamp_sec": round((frame_idx - 1) / fps, 3),
                "path_valid": res.perception.path_valid,
                "state": res.decision.state,
                "normalized_error": round(res.perception.normalized_error, 4),
                "path_center_x": round(res.perception.path_center_x, 1) if res.perception.path_center_x is not None else None,
                "left_pwm": res.decision.left_pwm,
                "right_pwm": res.decision.right_pwm,
                "processing_ms": round(res.processing_ms, 2),
                "fps": round(res.fps, 1),
                "reason": res.decision.reason,
            }
            records.append(rec)

            if progress_callback is not None and total_frames > 0:
                fraction = min(1.0, frame_idx / total_frames)
                progress_callback(fraction, frame_idx, total_frames)

    finally:
        cap.release()
        annotated_writer.release()
        mask_writer.release()
        side_by_side_writer.release()
        logger.close()

    duration = total_frames / fps if fps > 0 else 0.0

    return VideoBatchExportResult(
        input_path=input_path,
        annotated_video_path=annotated_path,
        mask_video_path=mask_path,
        side_by_side_video_path=side_by_side_path,
        csv_log_path=logger.log_path,
        total_frames=frame_idx,
        fps=fps,
        duration_sec=round(duration, 2),
        records=records,
    )
