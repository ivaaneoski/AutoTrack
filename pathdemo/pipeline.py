"""End-to-end vision and steering pipeline processing individual frames."""

import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from pathdemo.config import AppConfig
from pathdemo.perception import process_path_frame, PerceptionResult
from pathdemo.controller import PathSteeringController, SteeringDecision
from pathdemo.telemetry import TelemetryLogger


@dataclass
class PipelineResult:
    """Consolidated outcome of processing a single camera frame."""
    source_name: str
    frame_index: int
    timestamp: float
    original_frame: np.ndarray
    annotated_frame: np.ndarray
    mask: np.ndarray
    roi_mask: np.ndarray
    perception: PerceptionResult
    decision: SteeringDecision
    processing_ms: float
    fps: float


class PathFollowingPipeline:
    """Coordinates camera frame perception, controller logic, and performance benchmarking."""

    def __init__(self, config: AppConfig, telemetry_logger: Optional[TelemetryLogger] = None):
        self.config = config
        self.controller = PathSteeringController(config.controller)
        self.telemetry_logger = telemetry_logger
        self.frame_index = 0
        self._last_perf_time: Optional[float] = None

    def reset(self) -> None:
        """Reset internal sequence states."""
        self.controller.reset()
        self.frame_index = 0
        self._last_perf_time = None

    def update_config(self, config: AppConfig) -> None:
        """Dynamically update perception and controller parameters."""
        self.config = config
        self.controller.update_config(config.controller)

    def process_frame(self, frame: np.ndarray, source_name: str = "stream") -> PipelineResult:
        """Executes the complete pipeline on a single image frame with high-resolution timing."""
        start_t = time.perf_counter()

        # Compute elapsed time dt for derivative control
        dt: Optional[float] = None
        if self._last_perf_time is not None:
            dt = start_t - self._last_perf_time
        self._last_perf_time = start_t

        # 1. Perception: HSV Segmentation & Geometry
        perception_res = process_path_frame(frame, self.config.perception)

        # 2. Control: PD Steering & Simulated PWM Proposal
        decision = self.controller.compute(
            path_valid=perception_res.path_valid,
            normalized_error=perception_res.normalized_error,
            dt=dt,
        )

        end_t = time.perf_counter()
        elapsed_sec = end_t - start_t
        processing_ms = elapsed_sec * 1000.0
        fps = (1.0 / elapsed_sec) if elapsed_sec > 1e-6 else 999.0

        self.frame_index += 1

        # 3. CSV Telemetry Logging if enabled
        if self.telemetry_logger is not None:
            self.telemetry_logger.log_record(
                source=source_name,
                frame_index=self.frame_index,
                path_valid=perception_res.path_valid,
                path_center_x=perception_res.path_center_x,
                normalized_error=perception_res.normalized_error,
                state=decision.state,
                left_pwm=decision.left_pwm,
                right_pwm=decision.right_pwm,
                processing_ms=processing_ms,
                fps=fps,
            )

        return PipelineResult(
            source_name=source_name,
            frame_index=self.frame_index,
            timestamp=start_t,
            original_frame=frame,
            annotated_frame=perception_res.annotated_frame,
            mask=perception_res.mask,
            roi_mask=perception_res.roi_mask,
            perception=perception_res,
            decision=decision,
            processing_ms=processing_ms,
            fps=fps,
        )
