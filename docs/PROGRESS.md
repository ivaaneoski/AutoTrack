# Project Implementation Progress & Acceptance Log

Project: Camera-Based Path Following + Live Dashboard (Review-I Software Prototype)

## Acceptance Gate Checklist

| Phase | Description | Status | Evidence / Notes |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Inspect environment & scaffold modular structure | **COMPLETE** | Python 3.11.4 64-bit, OpenCV 5.0.0, NumPy 2.4.6, Streamlit 1.66.1 installed in `.venv`. Config loading & validation verified with 4 passing pytest tests. Git initialized. |
| **Phase 1** | Synthetic frames & OpenCV HSV path perception | **COMPLETE** | Synthetic generators implemented (straight, curve left/right, offset, noisy, missing). OpenCV HSV color pipeline and moment analysis passing 16 pytest tests. Output figures saved to `results/figures/`. |
| **Phase 2** | PD Steering controller & end-to-end pipeline | **COMPLETE** | Proportional/PD steering controller with differential PWM mixing `[0, 255]`, state transitions (`STRAIGHT`, `LEFT`, `RIGHT`, `PATH_LOST`), monotonic timer benchmarking, and CSV telemetry logger. Passing 25 tests. |
| **Phase 3** | Interactive Streamlit dashboard (Synthetic mode) | **COMPLETE** | Interactive Streamlit app (`app.py`) verified in browser with live telemetry metrics cards, annotated views, segmentation mask, rolling error chart, and calibration preset persistence. |
| **Phase 4** | Recorded video (Samsung S22 workflow) & Webcam source | **COMPLETE** | S22 workflow supported via direct video selection/upload, playback/reset controls, format validation, demo track MP4 generator, and robust webcam fallback when sensor is unavailable. |
| **Phase 5** | Acceptance testing, real output captures & documentation | **COMPLETE** | 30/30 pytest tests passing in 0.44s. Real UI screenshots in `results/figures/`, 150-frame video telemetry CSV in `results/logs/`, comprehensive README, technical explainer & viva defense guide complete. |

---

## Detailed Log

### Phase 0: Environment Inspection & Scaffolding
- Initialized local Git repository.
- Checked Python runtime: Python 3.11.4 64-bit on Windows.
- Created `.venv` and verified clean installation of `opencv-python==5.0.0.93`, `numpy==2.4.6`, `streamlit==1.66.1`, `pytest==9.1.1`, `pyyaml==6.0.3`, `pandas==3.1.0`, `pillow==13.0.0`.
- Documented versions in `docs/VERSIONS.md`.
- Scaffolded folder layout: `configs/`, `pathdemo/`, `tests/`, `tools/`, `data/input_videos/`, `results/logs/`, `results/figures/`, `docs/`.
- Created `.gitignore` protecting user media, caches, and logs.
- Config dataclasses and validation verified with 4 passing pytest tests.

### Phase 1: Synthetic Test Frames & OpenCV HSV Perception
- Implemented `pathdemo/synthetic.py` supporting 7 deterministic scenarios: `straight_centered`, `straight_left`, `straight_right`, `curve_left`, `curve_right`, `noisy_centered`, and `missing_path`.
- Implemented `pathdemo/perception.py`: BGR to HSV conversion, inRange thresholding, morphological open/close, ROI lower road cropping (50%-100%), minimum contour area filtering (400 px), spatial moment centroid calculation $(c_x, c_y)$, and normalized lateral error $[-1.0, +1.0]$.
- Verified noise rejection and geometric invariance across image resolutions.
- 16 pytest tests passing. Generated visual figures in `results/figures/`.

### Phase 2: PD Steering Controller & End-to-End Pipeline
- Implemented `pathdemo/controller.py`: Proportional and Derivative steering controller with differential wheel PWM mixing clamped to `[0, 255]`.
- Implemented states: `STRAIGHT`, `LEFT`, `RIGHT`, `PATH_LOST`, `STOP`. Verified instant stop (0 PWM) on path disappearance.
- Implemented `pathdemo/pipeline.py` with high-resolution monotonic timer (`time.perf_counter()`) for latency and FPS measurement.
- Implemented `pathdemo/telemetry.py` for structured CSV logging.
- 25 pytest tests passing.

### Phase 3: Interactive Streamlit Dashboard
- Implemented `app.py` featuring a clean, responsive layout:
  - Simulation notice banner: `SIMULATION / LAPTOP PROTOTYPE — NO HARDWARE CONTROL`.
  - Dynamic three-panel visualization: Annotated Road View, Segmentation Mask, Raw Camera Frame.
  - Six real-time telemetry metric cards: State, Lateral Error, Left PWM, Right PWM, Latency (ms), Throughput (FPS).
  - Rolling history charts for lateral error and motor PWM.
  - Full perception and controller tuning sliders with calibration preset saving/loading.
- Verified live interaction in browser subagent. Saved actual UI screenshots to `results/figures/dashboard_synthetic_centered.png` and `dashboard_synthetic_left.png`.

### Phase 4: Recorded Video (Samsung S22) & Webcam Modes
- Implemented `pathdemo/sources.py` unifying Synthetic, Video File, and Webcam input under `BaseFrameSource`.
- Supported Samsung Galaxy S22 workflow via video upload and direct local file playback.
- Created `tools/make_demo_video.py` producing `data/input_videos/sample_track.mp4` (150 frames, 30 FPS).
- Added graceful webcam fallback displaying actionable warnings rather than unhandled exceptions.
- Executed batch run `tools/run_video_pipeline_batch.py` processing all 150 frames and logging genuine telemetry to `results/logs/telemetry_video_demo_*.csv`.
- Captured milestone frames: `video_sample_straight`, `video_sample_curve_left`, `video_sample_curve_right`, `video_sample_gap_stop`.

### Phase 5: Acceptance Verification & Documentation
- Ran full automated pytest suite: **30 / 30 tests passing**.
- Authored `README.md` with complete installation, run, test, and S22 recording guidelines.
- Authored `docs/EXPLAINER.md` providing in-depth mathematical justifications, color theory explanations, and Viva review defense Q&A.
- Expanded `docs/KNOWN_ISSUES.md` detailing camera white balance shifts and video codec considerations.
