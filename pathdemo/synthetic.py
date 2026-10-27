"""Deterministic synthetic test image and clip generator for path following."""

from dataclasses import dataclass
from typing import Optional, Generator, Tuple
import numpy as np
import cv2


@dataclass
class SyntheticFrame:
    """Represents a generated synthetic camera frame and ground truth metadata."""
    frame: np.ndarray
    case_name: str
    has_path: bool
    ground_truth_center_x: Optional[float]
    ground_truth_error: Optional[float]
    width: int
    height: int


# Default high-contrast yellow color in BGR (Hue ~28 in OpenCV HSV)
BGR_YELLOW_PATH = (20, 215, 250)
BGR_ROAD_BACKGROUND = (65, 65, 65)
BGR_NOISE_BLOB = (180, 180, 180)


def create_blank_road(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a realistic asphalt-like textured background."""
    base = np.full((height, width, 3), BGR_ROAD_BACKGROUND, dtype=np.uint8)
    # Add subtle stationary texture without modifying deterministic output
    return base


def draw_path_ribbon(
    image: np.ndarray,
    center_points: np.ndarray,
    thickness: int = 28,
    color: Tuple[int, int, int] = BGR_YELLOW_PATH,
) -> None:
    """Draw an antialiased polyline ribbon representing the path."""
    pts = np.asarray(center_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def generate_synthetic_case(
    case_name: str,
    width: int = 640,
    height: int = 480,
    thickness: int = 28,
    seed: int = 42,
) -> SyntheticFrame:
    """Generate a single deterministic synthetic test frame for unit testing and demo.

    Supported case_name values:
        - "straight_centered": straight path down the optical axis
        - "straight_left": straight path offset to the left of center (negative error)
        - "straight_right": straight path offset to the right of center (positive error)
        - "curve_left": path sweeping to the left
        - "curve_right": path sweeping to the right
        - "noisy_centered": centered path with non-yellow distractors and small noise
        - "missing_path": road surface with no coloured path (loss of path)
    """
    frame = create_blank_road(width, height)
    image_center_x = width / 2.0
    y_coords = np.linspace(0, height, num=20)
    
    # Ground-truth evaluation occurs in the lower ROI (y = 0.75 * height)
    eval_y = 0.75 * height

    if case_name == "straight_centered":
        cx = image_center_x
        pts = np.column_stack([np.full_like(y_coords, cx), y_coords])
        draw_path_ribbon(frame, pts, thickness=thickness)
        err = 0.0
        return SyntheticFrame(
            frame=frame,
            case_name=case_name,
            has_path=True,
            ground_truth_center_x=float(cx),
            ground_truth_error=float(err),
            width=width,
            height=height,
        )

    elif case_name == "straight_left":
        # Offset path 120 pixels to the left
        cx = image_center_x - 140.0
        pts = np.column_stack([np.full_like(y_coords, cx), y_coords])
        draw_path_ribbon(frame, pts, thickness=thickness)
        err = (cx - image_center_x) / (width / 2.0)
        return SyntheticFrame(
            frame=frame,
            case_name=case_name,
            has_path=True,
            ground_truth_center_x=float(cx),
            ground_truth_error=float(err),
            width=width,
            height=height,
        )

    elif case_name == "straight_right":
        # Offset path 120 pixels to the right
        cx = image_center_x + 140.0
        pts = np.column_stack([np.full_like(y_coords, cx), y_coords])
        draw_path_ribbon(frame, pts, thickness=thickness)
        err = (cx - image_center_x) / (width / 2.0)
        return SyntheticFrame(
            frame=frame,
            case_name=case_name,
            has_path=True,
            ground_truth_center_x=float(cx),
            ground_truth_error=float(err),
            width=width,
            height=height,
        )

    elif case_name == "curve_left":
        # Quadratic curve arching left towards top
        # x(y) = cx + a * (y - H)^2
        # At bottom (y=height), near center; as y goes towards 0, x shifts left
        x_coords = image_center_x - 220.0 * ((height - y_coords) / height) ** 1.8
        pts = np.column_stack([x_coords, y_coords])
        draw_path_ribbon(frame, pts, thickness=thickness)
        gt_x = float(image_center_x - 220.0 * ((height - eval_y) / height) ** 1.8)
        err = (gt_x - image_center_x) / (width / 2.0)
        return SyntheticFrame(
            frame=frame,
            case_name=case_name,
            has_path=True,
            ground_truth_center_x=gt_x,
            ground_truth_error=err,
            width=width,
            height=height,
        )

    elif case_name == "curve_right":
        # Quadratic curve arching right towards top
        x_coords = image_center_x + 220.0 * ((height - y_coords) / height) ** 1.8
        pts = np.column_stack([x_coords, y_coords])
        draw_path_ribbon(frame, pts, thickness=thickness)
        gt_x = float(image_center_x + 220.0 * ((height - eval_y) / height) ** 1.8)
        err = (gt_x - image_center_x) / (width / 2.0)
        return SyntheticFrame(
            frame=frame,
            case_name=case_name,
            has_path=True,
            ground_truth_center_x=gt_x,
            ground_truth_error=err,
            width=width,
            height=height,
        )

    elif case_name == "noisy_centered":
        # Centered line + non-yellow distractors and small yellow speckles
        rng = np.random.default_rng(seed)
        cx = image_center_x
        pts = np.column_stack([np.full_like(y_coords, cx), y_coords])
        draw_path_ribbon(frame, pts, thickness=thickness)

        # Add small sub-threshold yellow speckles (area < 100 px each)
        for _ in range(8):
            rx = int(rng.integers(30, width - 30))
            ry = int(rng.integers(int(height * 0.5), height - 20))
            cv2.circle(frame, (rx, ry), radius=int(rng.integers(2, 5)), color=BGR_YELLOW_PATH, thickness=-1)

        # Add white/grey larger distractors (not in yellow HSV range)
        for _ in range(5):
            rx = int(rng.integers(30, width - 30))
            ry = int(rng.integers(int(height * 0.5), height - 20))
            cv2.rectangle(frame, (rx, ry), (rx + 25, ry + 25), color=BGR_NOISE_BLOB, thickness=-1)

        err = 0.0
        return SyntheticFrame(
            frame=frame,
            case_name=case_name,
            has_path=True,
            ground_truth_center_x=float(cx),
            ground_truth_error=float(err),
            width=width,
            height=height,
        )

    elif case_name == "missing_path":
        # Clean road surface without any path
        # Optionally add a grey non-path object
        cv2.rectangle(frame, (100, 300), (160, 340), color=BGR_NOISE_BLOB, thickness=-1)
        return SyntheticFrame(
            frame=frame,
            case_name=case_name,
            has_path=False,
            ground_truth_center_x=None,
            ground_truth_error=None,
            width=width,
            height=height,
        )

    else:
        raise ValueError(f"Unknown synthetic case: {case_name}")


def generate_synthetic_video_sequence(
    num_frames: int = 150,
    width: int = 640,
    height: int = 480,
    thickness: int = 28,
) -> Generator[np.ndarray, None, None]:
    """Generates a smooth continuous synthetic video stream through various path geometries:
    - 0 to 30: Straight centered
    - 31 to 65: Gentle curve left
    - 66 to 100: Gentle curve right
    - 101 to 125: Path missing (safety stop scenario)
    - 126 to 150: Re-acquired centered path
    """
    image_center_x = width / 2.0
    y_coords = np.linspace(0, height, num=25)

    for i in range(num_frames):
        frame = create_blank_road(width, height)
        
        if 0 <= i <= 30:
            # Straight path
            pts = np.column_stack([np.full_like(y_coords, image_center_x), y_coords])
            draw_path_ribbon(frame, pts, thickness=thickness)
        elif 31 <= i <= 65:
            # Transition to left curve
            progress = (i - 30) / 35.0
            shift = -180.0 * np.sin(progress * np.pi / 2.0)
            x_coords = image_center_x + shift * ((height - y_coords) / height) ** 1.6
            pts = np.column_stack([x_coords, y_coords])
            draw_path_ribbon(frame, pts, thickness=thickness)
        elif 66 <= i <= 100:
            # Transition from left to right curve
            progress = (i - 65) / 35.0
            shift = -180.0 + 360.0 * np.sin(progress * np.pi / 2.0)
            x_coords = image_center_x + shift * ((height - y_coords) / height) ** 1.6
            pts = np.column_stack([x_coords, y_coords])
            draw_path_ribbon(frame, pts, thickness=thickness)
        elif 101 <= i <= 125:
            # Missing path scenario (track break)
            pass
        else:
            # Reacquired straight path
            pts = np.column_stack([np.full_like(y_coords, image_center_x), y_coords])
            draw_path_ribbon(frame, pts, thickness=thickness)

        yield frame
