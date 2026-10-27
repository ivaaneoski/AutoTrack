"""Tests for the pure Python PD steering controller."""

import pytest
import numpy as np
from pathdemo.config import ControllerConfig
from pathdemo.controller import PathSteeringController


@pytest.fixture
def controller():
    cfg = ControllerConfig(
        kp=140.0,
        kd=30.0,
        deadband=0.05,
        base_speed=140,
        min_pwm=0,
        max_pwm=255,
    )
    return PathSteeringController(cfg)


def test_controller_centered_path_straight(controller):
    dec = controller.compute(path_valid=True, normalized_error=0.01, dt=0.033)
    assert dec.state == "STRAIGHT"
    assert dec.left_pwm == 140
    assert dec.right_pwm == 140
    assert dec.steering_command == 0.0


def test_controller_left_path_steers_left(controller):
    # Error < -deadband means path is to the left
    # To steer left, left wheel must decelerate (< base_speed) and right wheel must accelerate (> base_speed)
    dec = controller.compute(path_valid=True, normalized_error=-0.35, dt=0.033)
    assert dec.state == "LEFT"
    assert dec.steering_command < 0
    assert dec.left_pwm < 140
    assert dec.right_pwm > 140


def test_controller_right_path_steers_right(controller):
    # Error > +deadband means path is to the right
    # To steer right, left wheel must accelerate (> base_speed) and right wheel must decelerate (< base_speed)
    dec = controller.compute(path_valid=True, normalized_error=0.35, dt=0.033)
    assert dec.state == "RIGHT"
    assert dec.steering_command > 0
    assert dec.left_pwm > 140
    assert dec.right_pwm < 140


def test_controller_pwm_bounds_saturation(controller):
    # Test extreme errors saturation within [0, 255]
    dec_max_left = controller.compute(path_valid=True, normalized_error=-1.0, dt=0.033)
    assert 0 <= dec_max_left.left_pwm <= 255
    assert 0 <= dec_max_left.right_pwm <= 255
    assert dec_max_left.left_pwm == 0  # 140 - 140 = 0 (or clamped at 0)
    assert dec_max_left.right_pwm == 255  # Clamped at 255

    dec_max_right = controller.compute(path_valid=True, normalized_error=1.0, dt=0.033)
    assert 0 <= dec_max_right.left_pwm <= 255
    assert 0 <= dec_max_right.right_pwm <= 255
    assert dec_max_right.left_pwm == 255
    assert dec_max_right.right_pwm == 0


def test_controller_path_lost_safety_stop(controller):
    # Missing path must produce immediate STOP with 0 PWM
    dec = controller.compute(path_valid=False, normalized_error=0.5, dt=0.033)
    assert dec.state == "PATH_LOST"
    assert dec.left_pwm == 0
    assert dec.right_pwm == 0
    assert dec.steering_command == 0.0


def test_controller_robust_against_invalid_dt(controller):
    # dt = 0, negative dt, None, inf must not produce NaN or crash
    for bad_dt in [0.0, -0.05, None, float("inf"), float("nan")]:
        dec = controller.compute(path_valid=True, normalized_error=0.2, dt=bad_dt)
        assert np.isfinite(dec.steering_command)
        assert not np.isnan(dec.steering_command)
        assert 0 <= dec.left_pwm <= 255
        assert 0 <= dec.right_pwm <= 255
