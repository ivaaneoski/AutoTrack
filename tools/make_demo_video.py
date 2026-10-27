"""Generates a synthetic MP4 demo video clip for testing video file input."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from pathdemo.synthetic import generate_synthetic_video_sequence


def create_demo_video(
    output_path: str = "data/input_videos/sample_track.mp4",
    num_frames: int = 150,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> Path:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path: {out_file}")

    print(f"Generating {num_frames} synthetic frames to {out_file}...")
    for frame in generate_synthetic_video_sequence(num_frames=num_frames, width=width, height=height):
        writer.write(frame)

    writer.release()
    print(f"Demo video saved successfully: {out_file} ({out_file.stat().st_size} bytes)")
    return out_file


if __name__ == "__main__":
    create_demo_video()
