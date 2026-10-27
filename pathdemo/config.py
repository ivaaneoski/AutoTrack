"""Configuration loading, validation, and serialization."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import yaml


@dataclass
class PerceptionConfig:
    hsv_lower: Tuple[int, int, int] = (18, 70, 70)
    hsv_upper: Tuple[int, int, int] = (38, 255, 255)
    morph_kernel_size: int = 5
    roi_top_fraction: float = 0.50
    roi_bottom_fraction: float = 1.00
    min_contour_area: float = 400.0

    def validate(self) -> None:
        if len(self.hsv_lower) != 3 or len(self.hsv_upper) != 3:
            raise ValueError("hsv_lower and hsv_upper must contain 3 elements (H, S, V).")
        for i, (low, high) in enumerate(zip(self.hsv_lower, self.hsv_upper)):
            if not (0 <= low <= high <= 255):
                raise ValueError(f"Invalid HSV bound at index {i}: lower={low}, upper={high}")
        if self.morph_kernel_size <= 0 or self.morph_kernel_size % 2 == 0:
            raise ValueError("morph_kernel_size must be a positive odd integer.")
        if not (0.0 <= self.roi_top_fraction < self.roi_bottom_fraction <= 1.0):
            raise ValueError("ROI bounds must satisfy: 0.0 <= roi_top_fraction < roi_bottom_fraction <= 1.0")
        if self.min_contour_area < 0:
            raise ValueError("min_contour_area must be non-negative.")


@dataclass
class ControllerConfig:
    kp: float = 140.0
    kd: float = 35.0
    deadband: float = 0.05
    base_speed: int = 140
    max_pwm: int = 255
    min_pwm: int = 0

    def validate(self) -> None:
        if self.kp < 0 or self.kd < 0:
            raise ValueError("Controller gains kp and kd must be non-negative.")
        if not (0.0 <= self.deadband < 1.0):
            raise ValueError("deadband must be in range [0.0, 1.0).")
        if not (0 <= self.min_pwm <= self.base_speed <= self.max_pwm <= 255):
            raise ValueError("PWM bounds must satisfy: 0 <= min_pwm <= base_speed <= max_pwm <= 255.")


@dataclass
class TelemetryConfig:
    log_dir: str = "results/logs"
    save_csv: bool = True

    def validate(self) -> None:
        if not self.log_dir:
            raise ValueError("log_dir cannot be empty.")


@dataclass
class AppConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    def validate(self) -> None:
        self.perception.validate()
        self.controller.validate()
        self.telemetry.validate()


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults if not found."""
    config = AppConfig()
    if config_path is None:
        config_path = Path("configs/default.yaml")
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "perception" in data:
            p_data = data["perception"]
            config.perception = PerceptionConfig(
                hsv_lower=tuple(p_data.get("hsv_lower", config.perception.hsv_lower)),
                hsv_upper=tuple(p_data.get("hsv_upper", config.perception.hsv_upper)),
                morph_kernel_size=int(p_data.get("morph_kernel_size", config.perception.morph_kernel_size)),
                roi_top_fraction=float(p_data.get("roi_top_fraction", config.perception.roi_top_fraction)),
                roi_bottom_fraction=float(p_data.get("roi_bottom_fraction", config.perception.roi_bottom_fraction)),
                min_contour_area=float(p_data.get("min_contour_area", config.perception.min_contour_area)),
            )

        if "controller" in data:
            c_data = data["controller"]
            config.controller = ControllerConfig(
                kp=float(c_data.get("kp", config.controller.kp)),
                kd=float(c_data.get("kd", config.controller.kd)),
                deadband=float(c_data.get("deadband", config.controller.deadband)),
                base_speed=int(c_data.get("base_speed", config.controller.base_speed)),
                max_pwm=int(c_data.get("max_pwm", config.controller.max_pwm)),
                min_pwm=int(c_data.get("min_pwm", config.controller.min_pwm)),
            )

        if "telemetry" in data:
            t_data = data["telemetry"]
            config.telemetry = TelemetryConfig(
                log_dir=str(t_data.get("log_dir", config.telemetry.log_dir)),
                save_csv=bool(t_data.get("save_csv", config.telemetry.save_csv)),
            )

    config.validate()
    return config


def save_config(config: AppConfig, save_path: str | Path) -> None:
    """Save configuration to a YAML file."""
    config.validate()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "perception": {
            "hsv_lower": list(config.perception.hsv_lower),
            "hsv_upper": list(config.perception.hsv_upper),
            "morph_kernel_size": config.perception.morph_kernel_size,
            "roi_top_fraction": config.perception.roi_top_fraction,
            "roi_bottom_fraction": config.perception.roi_bottom_fraction,
            "min_contour_area": config.perception.min_contour_area,
        },
        "controller": {
            "kp": config.controller.kp,
            "kd": config.controller.kd,
            "deadband": config.controller.deadband,
            "base_speed": config.controller.base_speed,
            "max_pwm": config.controller.max_pwm,
            "min_pwm": config.controller.min_pwm,
        },
        "telemetry": {
            "log_dir": config.telemetry.log_dir,
            "save_csv": config.telemetry.save_csv,
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
