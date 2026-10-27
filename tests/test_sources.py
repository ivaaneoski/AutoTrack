"""Tests for frame input sources and safe fallbacks."""

import pytest
from pathlib import Path
from pathdemo.sources import SyntheticCaseSource, SyntheticStreamSource, VideoFileSource, WebcamSource


def test_synthetic_case_source():
    src = SyntheticCaseSource(case_name="straight_left", width=320, height=240)
    assert src.is_opened is True

    ok, frame, msg = src.read_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    assert "straight_left" in msg

    src.set_case("missing_path")
    ok, frame, msg = src.read_frame()
    assert ok is True
    assert "missing_path" in msg


def test_synthetic_stream_source():
    src = SyntheticStreamSource(num_frames=10, width=160, height=120, loop=False)
    assert src.is_opened is True

    count = 0
    while True:
        ok, frame, msg = src.read_frame()
        if not ok:
            break
        count += 1
        assert frame.shape == (120, 160, 3)

    assert count == 10

    # Next read should report completed
    ok, frame, msg = src.read_frame()
    assert ok is False
    assert "completed" in msg

    # Reset
    src.reset()
    ok, frame, msg = src.read_frame()
    assert ok is True


def test_video_file_source_read_and_reset():
    video_path = Path("data/input_videos/sample_track.mp4")
    assert video_path.exists(), "Sample video should exist for source tests"

    src = VideoFileSource(video_path, loop=False)
    assert src.is_opened is True
    assert src.total_frames > 0

    ok, frame, msg = src.read_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert src.current_frame == 1

    src.reset()
    assert src.current_frame == 0
    src.release()
    assert src.is_opened is False


def test_video_file_source_missing_file():
    src = VideoFileSource("data/input_videos/non_existent_file.mp4")
    assert src.is_opened is False
    ok, frame, msg = src.read_frame()
    assert ok is False
    assert "not found" in msg.lower()


def test_webcam_source_safe_fallback():
    # Index 999 is guaranteed invalid on standard laptop
    src = WebcamSource(device_index=999)
    assert src.is_opened is False
    ok, frame, msg = src.read_frame()
    assert ok is False
    assert "unavailable" in msg.lower() or "could not be opened" in msg.lower()
    src.release()
