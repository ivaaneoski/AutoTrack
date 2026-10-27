"""Video input sources: Synthetic, Video File, and Webcam."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2

from pathdemo.synthetic import generate_synthetic_case, generate_synthetic_video_sequence


class BaseFrameSource(ABC):
    """Abstract base class for all frame providers."""

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], str]:
        """Read the next frame.

        Returns:
            (success, frame_bgr, status_message)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Rewind or reset source to starting state."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Cleanly free underlying handles/resources."""
        pass

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is operational."""
        pass


class SyntheticCaseSource(BaseFrameSource):
    """Provides a single deterministic synthetic test scenario."""

    def __init__(self, case_name: str = "straight_centered", width: int = 640, height: int = 480):
        self.case_name = case_name
        self.width = width
        self.height = height
        self._frame_obj = generate_synthetic_case(case_name, width=width, height=height)

    def set_case(self, case_name: str) -> None:
        self.case_name = case_name
        self._frame_obj = generate_synthetic_case(case_name, width=self.width, height=self.height)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], str]:
        return True, self._frame_obj.frame.copy(), f"Synthetic Case: {self.case_name}"

    def reset(self) -> None:
        pass

    def release(self) -> None:
        pass

    @property
    def is_opened(self) -> bool:
        return True


class SyntheticStreamSource(BaseFrameSource):
    """Streams a continuous multi-segment synthetic driving sequence."""

    def __init__(self, num_frames: int = 150, width: int = 640, height: int = 480, loop: bool = True):
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.loop = loop
        self._frames = list(generate_synthetic_video_sequence(num_frames=num_frames, width=width, height=height))
        self._index = 0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], str]:
        if not self._frames:
            return False, None, "Synthetic stream is empty."

        if self._index >= len(self._frames):
            if self.loop:
                self._index = 0
            else:
                return False, None, "Synthetic stream completed."

        frame = self._frames[self._index].copy()
        msg = f"Synthetic Stream: frame {self._index + 1}/{len(self._frames)}"
        self._index += 1
        return True, frame, msg

    def reset(self) -> None:
        self._index = 0

    def release(self) -> None:
        self._frames.clear()

    @property
    def is_opened(self) -> bool:
        return len(self._frames) > 0


class VideoFileSource(BaseFrameSource):
    """Reads recorded video files (MP4, MOV, AVI, M4V)."""

    def __init__(self, file_path: str | Path, loop: bool = False):
        self.file_path = Path(file_path)
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.current_frame = 0
        self._open_file()

    def _open_file(self) -> None:
        if not self.file_path.exists():
            self.cap = None
            return

        self.cap = cv2.VideoCapture(str(self.file_path))
        if self.cap.isOpened():
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.current_frame = 0
        else:
            self.cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], str]:
        if not self.file_path.exists():
            return False, None, f"Video file not found: {self.file_path.name}"

        if self.cap is None or not self.cap.isOpened():
            return False, None, f"Unable to decode video file: {self.file_path.name}. Check format/codec."

        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.loop and self.total_frames > 0:
                self.reset()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return False, None, "End of video reached (loop rewind failed)."
            else:
                return False, None, "End of video reached."

        self.current_frame += 1
        return True, frame, f"Video File: {self.file_path.name} ({self.current_frame}/{max(1, self.total_frames)})"

    def reset(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


class WebcamSource(BaseFrameSource):
    """Direct video capture from a local USB/laptop webcam with safe fallback."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.cap: Optional[cv2.VideoCapture] = None
        self._error_msg: str = ""
        self._init_device()

    def _init_device(self) -> None:
        try:
            # Under Windows, CAP_DSHOW or default works
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                self._error_msg = f"Webcam device index {self.device_index} could not be opened."
                self.cap = None
                return

            # Test-read one frame to verify hardware permission and active sensor
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self._error_msg = f"Webcam device {self.device_index} opened but returned no frame."
                self.cap.release()
                self.cap = None
            else:
                self._error_msg = ""
        except Exception as e:
            self._error_msg = f"Exception opening webcam device {self.device_index}: {e}"
            self.cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], str]:
        if self.cap is None or not self.cap.isOpened():
            return False, None, self._error_msg or f"Webcam {self.device_index} is unavailable."

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, "Failed to capture frame from webcam."

        return True, frame, f"Live Webcam (Device {self.device_index})"

    def reset(self) -> None:
        pass

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
