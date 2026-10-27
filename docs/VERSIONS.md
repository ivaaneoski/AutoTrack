# Verified Environment & Dependency Versions

This document records the exact runtime environment and installed package versions verified on the development system.

## Operating System & Hardware
- **Operating System:** Windows 64-bit (Windows 11 / 10)
- **CPU:** x86_64 CPU (CPU-only execution, integrated graphics, no CUDA or GPU acceleration required)
- **Python Runtime:** Python 3.11.4 64-bit (`C:\Python37\python.exe` / `.venv`)

## Installed Package Versions (Verified Clean Installation)

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `opencv-python` | `5.0.0.93` | Computer vision pipeline: image/video capture, colour conversion, HSV thresholding, morphology, contour analysis, visualization |
| `numpy` | `2.4.6` | Numerical processing, bounding boxes, vector mathematics |
| `streamlit` | `1.66.1` | Local interactive web dashboard for real-time visualization, HSV slider tuning, CSV log triggering |
| `pytest` | `9.1.1` | Automated test suite for unit, integration, and safety acceptance gates |
| `pyyaml` | `6.0.3` | Reading and persisting detector and controller configurations (`configs/*.yaml`) |
| `pandas` | `3.1.0` | Streamlit telemetry data tables and charting utilities |
| `pillow` | `13.0.0` | Image handling and Streamlit visual components |

## Notes on Version Selection
- Standard CPU-only operation was verified without requiring dedicated GPU or external cloud services.
- OpenCV 5.0.0 and NumPy 2.4.6 installed with pre-built binary wheels for Python 3.11 Windows AMD64.
- No training frameworks (PyTorch, TensorFlow, YOLO) are installed or needed for the classic CV baseline.
