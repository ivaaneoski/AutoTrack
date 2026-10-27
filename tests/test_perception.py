"""Tests for OpenCV-based path perception module."""

import pytest
import numpy as np
import cv2
from pathdemo.config import PerceptionConfig
from pathdemo.perception import process_path_frame
from pathdemo.synthetic import generate_synthetic_case, BGR_YELLOW_PATH


@pytest.fixture
def default_config() -> PerceptionConfig:
    return PerceptionConfig()


def test_perception_centered_path(default_config):
    syn = generate_synthetic_case("straight_centered", width=640, height=480)
    res = process_path_frame(syn.frame, default_config)

    assert res.path_valid is True
    assert res.path_center_x is not None
    assert abs(res.normalized_error) < 0.08
    assert abs(res.path_center_x - 320.0) < 25.0
    assert res.contour_area > default_config.min_contour_area
    assert res.mask.shape == (480, 640)
    assert res.annotated_frame.shape == (480, 640, 3)


def test_perception_left_offset_path(default_config):
    syn = generate_synthetic_case("straight_left", width=640, height=480)
    res = process_path_frame(syn.frame, default_config)

    assert res.path_valid is True
    assert res.path_center_x is not None
    assert res.normalized_error < -0.2
    assert res.path_center_x < 320.0


def test_perception_right_offset_path(default_config):
    syn = generate_synthetic_case("straight_right", width=640, height=480)
    res = process_path_frame(syn.frame, default_config)

    assert res.path_valid is True
    assert res.path_center_x is not None
    assert res.normalized_error > 0.2
    assert res.path_center_x > 320.0


def test_perception_curved_paths(default_config):
    syn_left = generate_synthetic_case("curve_left", width=640, height=480)
    res_left = process_path_frame(syn_left.frame, default_config)
    assert res_left.path_valid is True
    assert res_left.normalized_error < 0.0

    syn_right = generate_synthetic_case("curve_right", width=640, height=480)
    res_right = process_path_frame(syn_right.frame, default_config)
    assert res_right.path_valid is True
    assert res_right.normalized_error > 0.0


def test_perception_missing_path_safety(default_config):
    syn = generate_synthetic_case("missing_path", width=640, height=480)
    res = process_path_frame(syn.frame, default_config)

    assert res.path_valid is False
    assert res.path_center_x is None
    assert res.path_center_y is None
    assert res.normalized_error == 0.0
    assert res.contour_area == 0.0
    assert "PATH_LOST" in res.notes


def test_perception_noise_rejection(default_config):
    # Create blank image with tiny yellow speckles (area < 50 px each)
    blank = np.full((480, 640, 3), (65, 65, 65), dtype=np.uint8)
    cv2.circle(blank, (320, 350), radius=3, color=BGR_YELLOW_PATH, thickness=-1)
    cv2.circle(blank, (200, 380), radius=4, color=BGR_YELLOW_PATH, thickness=-1)

    res = process_path_frame(blank, default_config)
    # The tiny blobs must be rejected because they are far below min_contour_area (400)
    assert res.path_valid is False
    assert res.path_center_x is None


def test_perception_arbitrary_dimensions(default_config):
    for w, h in [(320, 240), (800, 600), (1280, 720)]:
        syn = generate_synthetic_case("straight_centered", width=w, height=h)
        res = process_path_frame(syn.frame, default_config)
        assert res.path_valid is True
        assert abs(res.normalized_error) < 0.1
        assert res.mask.shape == (h, w)
        assert res.annotated_frame.shape == (h, w, 3)


def test_perception_invalid_inputs(default_config):
    with pytest.raises(ValueError):
        process_path_frame(None, default_config)

    with pytest.raises(ValueError):
        process_path_frame(np.zeros((100, 100), dtype=np.uint8), default_config)  # 2D instead of 3D
