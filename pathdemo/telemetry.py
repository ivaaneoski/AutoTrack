"""CSV Telemetry logger for path following prototype."""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class TelemetryLogger:
    """Manages real-time CSV logging of perception and controller metrics."""

    FIELDNAMES = [
        "timestamp",
        "iso_time",
        "source",
        "frame_index",
        "path_valid",
        "path_center_x",
        "normalized_error",
        "state",
        "left_pwm",
        "right_pwm",
        "processing_ms",
        "fps",
    ]

    def __init__(self, log_dir: str = "results/logs", prefix: str = "run"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"telemetry_{prefix}_{timestamp_str}.csv"
        self._file = open(self.log_path, mode="w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()
        self.records_logged = 0

    def log_record(
        self,
        source: str,
        frame_index: int,
        path_valid: bool,
        path_center_x: Optional[float],
        normalized_error: float,
        state: str,
        left_pwm: int,
        right_pwm: int,
        processing_ms: float,
        fps: float,
    ) -> Dict[str, Any]:
        """Logs a single telemetry record and flushes to disk."""
        now = time.time()
        record = {
            "timestamp": f"{now:.4f}",
            "iso_time": datetime.now().isoformat(),
            "source": source,
            "frame_index": frame_index,
            "path_valid": bool(path_valid),
            "path_center_x": f"{path_center_x:.2f}" if path_center_x is not None else "",
            "normalized_error": f"{normalized_error:+.4f}",
            "state": state,
            "left_pwm": int(left_pwm),
            "right_pwm": int(right_pwm),
            "processing_ms": f"{processing_ms:.2f}",
            "fps": f"{fps:.2f}",
        }
        self._writer.writerow(record)
        self._file.flush()
        self.records_logged += 1
        return record

    def close(self) -> None:
        """Safely flush and close the CSV log file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
