"""Tests for end-to-end pipeline and telemetry logging."""

import csv
from pathlib import Path
import pytest
from pathdemo.config import AppConfig
from pathdemo.pipeline import PathFollowingPipeline
from pathdemo.synthetic import generate_synthetic_case
from pathdemo.telemetry import TelemetryLogger


def test_pipeline_end_to_end_single_frame():
    config = AppConfig()
    pipeline = PathFollowingPipeline(config)

    syn = generate_synthetic_case("straight_centered", width=640, height=480)
    res = pipeline.process_frame(syn.frame, source_name="unit_test")

    assert res.frame_index == 1
    assert res.source_name == "unit_test"
    assert res.perception.path_valid is True
    assert res.decision.state == "STRAIGHT"
    assert res.decision.left_pwm == 140
    assert res.decision.right_pwm == 140
    assert res.processing_ms > 0
    assert res.fps > 0


def test_pipeline_sequence_state_transitions():
    config = AppConfig()
    pipeline = PathFollowingPipeline(config)

    # 1. Centered -> STRAIGHT
    res1 = pipeline.process_frame(generate_synthetic_case("straight_centered").frame)
    assert res1.decision.state == "STRAIGHT"

    # 2. Left offset -> LEFT
    res2 = pipeline.process_frame(generate_synthetic_case("straight_left").frame)
    assert res2.decision.state == "LEFT"
    assert res2.decision.left_pwm < res2.decision.right_pwm

    # 3. Right offset -> RIGHT
    res3 = pipeline.process_frame(generate_synthetic_case("straight_right").frame)
    assert res3.decision.state == "RIGHT"
    assert res3.decision.left_pwm > res3.decision.right_pwm

    # 4. Missing path -> PATH_LOST with zero PWM
    res4 = pipeline.process_frame(generate_synthetic_case("missing_path").frame)
    assert res4.decision.state == "PATH_LOST"
    assert res4.decision.left_pwm == 0
    assert res4.decision.right_pwm == 0


def test_telemetry_csv_logging(tmp_path: Path):
    log_dir = tmp_path / "logs"
    logger = TelemetryLogger(log_dir=str(log_dir), prefix="pytest")

    config = AppConfig()
    pipeline = PathFollowingPipeline(config, telemetry_logger=logger)

    # Process 3 frames
    pipeline.process_frame(generate_synthetic_case("straight_centered").frame, source_name="syn1")
    pipeline.process_frame(generate_synthetic_case("straight_left").frame, source_name="syn2")
    pipeline.process_frame(generate_synthetic_case("missing_path").frame, source_name="syn3")

    logger.close()

    # Verify CSV file contents
    csv_file = logger.log_path
    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 3
    assert reader[0]["source"] == "syn1"
    assert reader[0]["state"] == "STRAIGHT"
    assert reader[0]["path_valid"] == "True"

    assert reader[1]["source"] == "syn2"
    assert reader[1]["state"] == "LEFT"

    assert reader[2]["source"] == "syn3"
    assert reader[2]["state"] == "PATH_LOST"
    assert reader[2]["path_valid"] == "False"
    assert reader[2]["left_pwm"] == "0"
    assert reader[2]["right_pwm"] == "0"
