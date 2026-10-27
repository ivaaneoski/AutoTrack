"""Tests for configuration loading, validation, and serialization."""

import pytest
from pathlib import Path
from pathdemo.config import AppConfig, PerceptionConfig, ControllerConfig, load_config, save_config


def test_default_config_loads_and_validates():
    config = load_config("configs/default.yaml")
    assert config is not None
    assert config.perception.morph_kernel_size == 5
    assert 0 <= config.perception.roi_top_fraction < config.perception.roi_bottom_fraction <= 1.0
    assert config.controller.base_speed == 140
    assert config.controller.kp > 0


def test_invalid_perception_config_raises():
    with pytest.raises(ValueError):
        p = PerceptionConfig(morph_kernel_size=4)  # Must be odd
        p.validate()

    with pytest.raises(ValueError):
        p = PerceptionConfig(roi_top_fraction=0.9, roi_bottom_fraction=0.2)  # top > bottom
        p.validate()

    with pytest.raises(ValueError):
        p = PerceptionConfig(hsv_lower=(50, 10, 10), hsv_upper=(40, 200, 200))  # lower > upper
        p.validate()


def test_invalid_controller_config_raises():
    with pytest.raises(ValueError):
        c = ControllerConfig(kp=-1.0)
        c.validate()

    with pytest.raises(ValueError):
        c = ControllerConfig(deadband=-0.1)
        c.validate()

    with pytest.raises(ValueError):
        c = ControllerConfig(base_speed=300)  # > max_pwm
        c.validate()


def test_save_and_reload_config(tmp_path: Path):
    temp_yaml = tmp_path / "custom_config.yaml"
    cfg = AppConfig()
    cfg.perception.min_contour_area = 750.0
    cfg.controller.kp = 180.0
    save_config(cfg, temp_yaml)

    loaded = load_config(temp_yaml)
    assert loaded.perception.min_contour_area == 750.0
    assert loaded.controller.kp == 180.0
