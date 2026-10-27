"""Tests for synthetic frame and clip generation."""

import pytest
import numpy as np
from pathdemo.synthetic import generate_synthetic_case, generate_synthetic_video_sequence


def test_synthetic_case_dimensions_and_types():
    for case in [
        "straight_centered",
        "straight_left",
        "straight_right",
        "curve_left",
        "curve_right",
        "noisy_centered",
        "missing_path",
    ]:
        syn = generate_synthetic_case(case, width=320, height=240)
        assert syn.frame.shape == (240, 320, 3)
        assert syn.frame.dtype == np.uint8
        assert syn.width == 320
        assert syn.height == 240
        if case == "missing_path":
            assert syn.has_path is False
            assert syn.ground_truth_center_x is None
            assert syn.ground_truth_error is None
        else:
            assert syn.has_path is True
            assert syn.ground_truth_center_x is not None


def test_synthetic_ground_truth_signs():
    syn_center = generate_synthetic_case("straight_centered", width=640, height=480)
    syn_left = generate_synthetic_case("straight_left", width=640, height=480)
    syn_right = generate_synthetic_case("straight_right", width=640, height=480)

    assert abs(syn_center.ground_truth_error) < 0.05
    assert syn_left.ground_truth_error < -0.2
    assert syn_right.ground_truth_error > 0.2


def test_synthetic_video_sequence_length_and_shapes():
    frames = list(generate_synthetic_video_sequence(num_frames=20, width=160, height=120))
    assert len(frames) == 20
    for f in frames:
        assert f.shape == (120, 160, 3)
        assert f.dtype == np.uint8


def test_unknown_synthetic_case_raises():
    with pytest.raises(ValueError):
        generate_synthetic_case("non_existent_case")
