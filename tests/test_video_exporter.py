"""Tests for batch video processing and video exporter."""

from pathlib import Path
import pytest
from pathdemo.config import AppConfig
from pathdemo.video_exporter import process_and_export_video


def test_batch_video_export(tmp_path: Path):
    video_input = Path("data/input_videos/sample_track.mp4")
    assert video_input.exists(), "Sample track must exist"

    cfg = AppConfig()
    progress_calls = []

    def on_progress(frac, cur, total):
        progress_calls.append((frac, cur, total))

    result = process_and_export_video(
        video_path=video_input,
        output_dir=tmp_path,
        config=cfg,
        progress_callback=on_progress,
    )

    assert result.total_frames == 150
    assert result.annotated_video_path.exists()
    assert result.annotated_video_path.stat().st_size > 0

    assert result.mask_video_path.exists()
    assert result.mask_video_path.stat().st_size > 0

    assert result.side_by_side_video_path.exists()
    assert result.side_by_side_video_path.stat().st_size > 0

    assert result.csv_log_path.exists()
    assert len(result.records) == 150
    assert len(progress_calls) == 150
