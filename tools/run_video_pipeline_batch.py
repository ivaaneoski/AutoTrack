"""Process sample video track end-to-end, generating CSV logs and real frame captures."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from pathdemo.config import load_config
from pathdemo.pipeline import PathFollowingPipeline
from pathdemo.sources import VideoFileSource
from pathdemo.telemetry import TelemetryLogger


def main():
    config = load_config("configs/default.yaml")
    logger = TelemetryLogger(log_dir="results/logs", prefix="video_demo")
    pipeline = PathFollowingPipeline(config, telemetry_logger=logger)

    video_source = VideoFileSource("data/input_videos/sample_track.mp4", loop=False)
    if not video_source.is_opened:
        print("Error: Could not open sample_track.mp4")
        return

    out_figures = Path("results/figures")
    out_figures.mkdir(parents=True, exist_ok=True)

    print(f"Processing video frames from {video_source.file_path.name}...")
    frame_count = 0
    saved_samples = 0

    while True:
        ok, frame, msg = video_source.read_frame()
        if not ok or frame is None:
            break

        res = pipeline.process_frame(frame, source_name="sample_track.mp4")
        frame_count += 1

        # Capture key milestone frames: straight, curve left, curve right, gap
        if frame_count in [15, 50, 85, 115]:
            label = {
                15: "video_sample_straight",
                50: "video_sample_curve_left",
                85: "video_sample_curve_right",
                115: "video_sample_gap_stop",
            }[frame_count]

            cv2.imwrite(str(out_figures / f"{label}_annotated.png"), res.annotated_frame)
            cv2.imwrite(str(out_figures / f"{label}_mask.png"), res.mask)
            saved_samples += 1
            print(f"Captured {label}: state={res.decision.state}, error={res.perception.normalized_error:+.3f}, left_pwm={res.decision.left_pwm}, right_pwm={res.decision.right_pwm}")

    video_source.release()
    logger.close()

    print(f"Video batch run completed: {frame_count} frames processed.")
    print(f"Telemetry CSV written to: {logger.log_path} ({logger.records_logged} records)")


if __name__ == "__main__":
    main()
