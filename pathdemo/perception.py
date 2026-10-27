"""OpenCV-based HSV color segmentation and path geometry estimation."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

from pathdemo.config import PerceptionConfig


@dataclass
class PerceptionResult:
    """Structured perception output representing road path state."""
    path_valid: bool
    path_center_x: Optional[float]
    path_center_y: Optional[float]
    normalized_error: float
    mask: np.ndarray
    roi_mask: np.ndarray
    annotated_frame: np.ndarray
    contour_area: float
    confidence_proxy: float
    notes: str


def process_path_frame(
    frame: np.ndarray,
    config: PerceptionConfig,
) -> PerceptionResult:
    """Processes a single BGR camera frame through the classic CV pipeline.

    Steps:
    1. Validate input frame dimensions.
    2. Convert BGR color space to HSV.
    3. Apply HSV color range thresholding to isolate target tape/ribbon.
    4. Apply morphological OPEN and CLOSE operations to filter noise speckles.
    5. Restrict path analysis to the configurable Region of Interest (ROI) in the lower frame.
    6. Identify connected contours and reject candidates below min_contour_area.
    7. Calculate path centroid (cx, cy) via spatial moments on the primary contour.
    8. Calculate normalized lateral error: (cx - W/2) / (W/2) in [-1.0, 1.0].
    9. Construct informative annotated visual overlay.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Invalid or empty input frame provided to perception pipeline.")

    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {frame.shape}")

    height, width, _ = frame.shape
    image_center_x = width / 2.0

    # 1. Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. HSV thresholding
    lower_bound = np.array(config.hsv_lower, dtype=np.uint8)
    upper_bound = np.array(config.hsv_upper, dtype=np.uint8)
    raw_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 3. Morphological noise filtering
    k_size = config.morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    # Opening removes small bright noise speckles; closing fills small internal holes
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Region of Interest (ROI) extraction
    roi_top = int(height * config.roi_top_fraction)
    roi_bottom = int(height * config.roi_bottom_fraction)
    roi_top = max(0, min(height - 1, roi_top))
    roi_bottom = max(roi_top + 1, min(height, roi_bottom))
    roi_height = roi_bottom - roi_top

    # Isolate ROI in the mask
    roi_mask = np.zeros_like(cleaned_mask)
    roi_mask[roi_top:roi_bottom, :] = cleaned_mask[roi_top:roi_bottom, :]
    roi_sub = cleaned_mask[roi_top:roi_bottom, :]

    # 5. Connected contour detection within ROI
    contours, _ = cv2.findContours(roi_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [c for c in contours if cv2.contourArea(c) >= config.min_contour_area]

    annotated = frame.copy()

    # Draw ROI boundaries (subtle cyan horizontal lines and rectangle)
    cv2.rectangle(
        annotated,
        (0, roi_top),
        (width - 1, roi_bottom - 1),
        color=(255, 200, 0),  # Cyan-blue boundary
        thickness=1,
    )
    cv2.putText(
        annotated,
        f"ROI [{int(config.roi_top_fraction*100)}%-{int(config.roi_bottom_fraction*100)}%]",
        (10, max(20, roi_top - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 200, 0),
        1,
        cv2.LINE_AA,
    )

    # Draw vertical reference line (frame center optical axis)
    cv2.line(
        annotated,
        (int(image_center_x), 0),
        (int(image_center_x), height),
        color=(120, 120, 120),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    if not valid_contours:
        # PATH LOST state: No component meets minimum area threshold
        notes = "PATH_LOST: No qualifying path contour found in ROI."
        
        # Red warning banner
        cv2.putText(
            annotated,
            "PATH LOST / NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Min Area Req: {int(config.min_contour_area)} px",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 200),
            1,
            cv2.LINE_AA,
        )

        return PerceptionResult(
            path_valid=False,
            path_center_x=None,
            path_center_y=None,
            normalized_error=0.0,
            mask=cleaned_mask,
            roi_mask=roi_mask,
            annotated_frame=annotated,
            contour_area=0.0,
            confidence_proxy=0.0,
            notes=notes,
        )

    # 6. Select primary path component (largest contour by area)
    largest_contour = max(valid_contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))

    # Calculate spatial moments on the contour (offset by roi_top to map back to full frame)
    M = cv2.moments(largest_contour)
    if M["m00"] <= 0:
        return PerceptionResult(
            path_valid=False,
            path_center_x=None,
            path_center_y=None,
            normalized_error=0.0,
            mask=cleaned_mask,
            roi_mask=roi_mask,
            annotated_frame=annotated,
            contour_area=area,
            confidence_proxy=0.0,
            notes="PATH_LOST: Contour zero-moment.",
        )

    cx_roi = M["m10"] / M["m00"]
    cy_roi = M["m01"] / M["m00"]

    cx = float(cx_roi)
    cy = float(cy_roi + roi_top)

    # 7. Normalized error: [-1.0, +1.0]
    # Negative = path is left of image center; Positive = path is right of image center
    error = (cx - image_center_x) / (width / 2.0)
    error = float(np.clip(error, -1.0, 1.0))

    # Confidence proxy: ratio of detected contour area to ROI area, capped at 1.0
    # Transparently documented as detected mask pixel coverage proxy
    expected_nominal_area = width * roi_height * 0.08  # Approx 8% road occupancy
    confidence_proxy = float(np.clip(area / max(1.0, expected_nominal_area), 0.05, 1.0))

    # 8. Render rich visual overlay annotations
    # Translate contour coordinates to full image frame for rendering
    full_frame_contour = largest_contour.copy()
    full_frame_contour[:, :, 1] += roi_top
    cv2.drawContours(annotated, [full_frame_contour], -1, color=(0, 255, 0), thickness=2)

    # Draw centroid crosshair and point
    cv2.circle(annotated, (int(cx), int(cy)), radius=7, color=(0, 0, 255), thickness=-1)
    cv2.drawMarker(
        annotated,
        (int(cx), int(cy)),
        color=(255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
    )

    # Draw steering guidance vector line from vehicle bottom-center to path centroid
    vehicle_origin = (int(image_center_x), height - 2)
    cv2.line(
        annotated,
        vehicle_origin,
        (int(cx), int(cy)),
        color=(0, 255, 255),  # Yellow vector line
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    # Onscreen telemetry badge
    status_text = f"PATH VALID | Error: {error:+.2f} | Area: {int(area)} px"
    cv2.putText(
        annotated,
        status_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Centroid: ({int(cx)}, {int(cy)}) | Confidence Proxy: {confidence_proxy*100:.0f}%",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )

    return PerceptionResult(
        path_valid=True,
        path_center_x=cx,
        path_center_y=cy,
        normalized_error=error,
        mask=cleaned_mask,
        roi_mask=roi_mask,
        annotated_frame=annotated,
        contour_area=area,
        confidence_proxy=confidence_proxy,
        notes="Path detected successfully.",
    )
