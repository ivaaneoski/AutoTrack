"""Streamlit Interactive Dashboard for Camera-Based Path Following Prototype.

SIMULATION / LAPTOP PROTOTYPE — NO HARDWARE CONTROL.
"""

import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

# Ensure repository root is on Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from pathdemo.config import AppConfig, PerceptionConfig, ControllerConfig, load_config, save_config
from pathdemo.pipeline import PathFollowingPipeline, PipelineResult
from pathdemo.sources import (
    BaseFrameSource,
    SyntheticCaseSource,
    SyntheticStreamSource,
    VideoFileSource,
    WebcamSource,
)
from pathdemo.telemetry import TelemetryLogger

# Page configuration
st.set_page_config(
    page_title="Path Following Prototype | Review-I",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for high-contrast telemetry indicators
st.markdown(
    """
    <style>
    .notice-banner {
        background-color: #2b1d0c;
        border-left: 5px solid #ff9800;
        padding: 10px 16px;
        margin-bottom: 15px;
        border-radius: 4px;
        color: #ffcc80;
        font-weight: 500;
    }
    .metric-card {
        background-color: #1a1c24;
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #2d3139;
        text-align: center;
    }
    .state-straight { color: #00e676; font-size: 24px; font-weight: bold; }
    .state-left { color: #29b6f6; font-size: 24px; font-weight: bold; }
    .state-right { color: #ffa726; font-size: 24px; font-weight: bold; }
    .state-lost { color: #ef5350; font-size: 24px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize persistent Streamlit session state variables."""
    if "config" not in st.session_state:
        st.session_state.config = load_config("configs/default.yaml")
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = PathFollowingPipeline(st.session_state.config)
    if "telemetry_logger" not in st.session_state:
        st.session_state.telemetry_logger = None
    if "logging_enabled" not in st.session_state:
        st.session_state.logging_enabled = False
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "active_source" not in st.session_state:
        st.session_state.active_source = None
    if "source_type" not in st.session_state:
        st.session_state.source_type = "Synthetic Demo"
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=80)
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Ready."


init_session_state()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🎮 System Configuration")

# Input Mode Selection
input_mode = st.sidebar.radio(
    "Select Input Source",
    options=["Synthetic Demo", "Video File", "Webcam"],
    index=["Synthetic Demo", "Video File", "Webcam"].index(st.session_state.source_type),
    help="Choose synthetic generation, phone recording (Samsung S22/MP4), or live webcam.",
)

if input_mode != st.session_state.source_type:
    # Source changed: release existing source cleanly
    if st.session_state.active_source is not None:
        st.session_state.active_source.release()
        st.session_state.active_source = None
    st.session_state.source_type = input_mode
    st.session_state.is_playing = False
    st.session_state.pipeline.reset()
    st.session_state.history.clear()

selected_case = None
video_path_to_use = None
webcam_idx = 0

if input_mode == "Synthetic Demo":
    synthetic_submode = st.sidebar.selectbox(
        "Synthetic Mode",
        options=["Static Test Case", "Streaming Sequence"],
    )
    if synthetic_submode == "Static Test Case":
        selected_case = st.sidebar.selectbox(
            "Test Scenario",
            options=[
                "straight_centered",
                "straight_left",
                "straight_right",
                "curve_left",
                "curve_right",
                "noisy_centered",
                "missing_path",
            ],
            index=0,
        )
elif input_mode == "Video File":
    st.sidebar.markdown("### 📁 Video Source")
    video_input_type = st.sidebar.radio("File Selection", ["Existing Video", "Upload New Video (S22)"])
    
    existing_videos = list(Path("data/input_videos").glob("*.*"))
    supported_exts = {".mp4", ".mov", ".avi", ".m4v"}
    existing_videos = [v for v in existing_videos if v.suffix.lower() in supported_exts]

    if video_input_type == "Existing Video":
        if existing_videos:
            video_choice = st.sidebar.selectbox(
                "Choose local video",
                options=[v.name for v in existing_videos],
            )
            video_path_to_use = Path("data/input_videos") / video_choice
        else:
            st.sidebar.info("No videos found in data/input_videos/. Run tools/make_demo_video.py or upload a video.")
            video_path_to_use = None
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload S22 Recording",
            type=["mp4", "mov", "avi", "m4v"],
            help="Upload clips recorded on phone (e.g., Samsung Galaxy S22).",
        )
        if uploaded_file is not None:
            save_dest = Path("data/input_videos") / uploaded_file.name
            with open(save_dest, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path_to_use = save_dest
            st.sidebar.success(f"Saved: {uploaded_file.name}")
else:
    webcam_idx = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)

# Perception Tuning (Sliders)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 HSV Thresholds & ROI")

h_col1, h_col2 = st.sidebar.columns(2)
h_low = h_col1.slider("Hue Min", 0, 179, int(st.session_state.config.perception.hsv_lower[0]))
h_high = h_col2.slider("Hue Max", 0, 179, int(st.session_state.config.perception.hsv_upper[0]))

s_col1, s_col2 = st.sidebar.columns(2)
s_low = s_col1.slider("Sat Min", 0, 255, int(st.session_state.config.perception.hsv_lower[1]))
s_high = s_col2.slider("Sat Max", 0, 255, int(st.session_state.config.perception.hsv_upper[1]))

v_col1, v_col2 = st.sidebar.columns(2)
v_low = v_col1.slider("Val Min", 0, 255, int(st.session_state.config.perception.hsv_lower[2]))
v_high = v_col2.slider("Val Max", 0, 255, int(st.session_state.config.perception.hsv_upper[2]))

morph_k = st.sidebar.select_slider(
    "Morphology Kernel",
    options=[1, 3, 5, 7, 9],
    value=st.session_state.config.perception.morph_kernel_size,
)

roi_range = st.sidebar.slider(
    "ROI Height Range (Top - Bottom %)",
    min_value=0.0,
    max_value=1.0,
    value=(
        float(st.session_state.config.perception.roi_top_fraction),
        float(st.session_state.config.perception.roi_bottom_fraction),
    ),
    step=0.05,
)

min_area = st.sidebar.slider(
    "Min Component Area (px)",
    min_value=50,
    max_value=3000,
    value=int(st.session_state.config.perception.min_contour_area),
    step=50,
)

# Update configuration object dynamically
st.session_state.config.perception.hsv_lower = (int(h_low), int(s_low), int(v_low))
st.session_state.config.perception.hsv_upper = (int(h_high), int(s_high), int(v_high))
st.session_state.config.perception.morph_kernel_size = int(morph_k)
st.session_state.config.perception.roi_top_fraction = float(roi_range[0])
st.session_state.config.perception.roi_bottom_fraction = float(roi_range[1])
st.session_state.config.perception.min_contour_area = float(min_area)

# Controller Tuning
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Steering Controller")
kp_val = st.sidebar.slider("Kp (Proportional Gain)", 0.0, 300.0, float(st.session_state.config.controller.kp), 5.0)
kd_val = st.sidebar.slider("Kd (Derivative Gain)", 0.0, 100.0, float(st.session_state.config.controller.kd), 2.0)
deadband_val = st.sidebar.slider("Deadband", 0.0, 0.20, float(st.session_state.config.controller.deadband), 0.01)
base_pwm = st.sidebar.slider("Base Speed PWM", 50, 255, int(st.session_state.config.controller.base_speed), 5)

st.session_state.config.controller.kp = float(kp_val)
st.session_state.config.controller.kd = float(kd_val)
st.session_state.config.controller.deadband = float(deadband_val)
st.session_state.config.controller.base_speed = int(base_pwm)
st.session_state.pipeline.update_config(st.session_state.config)

# Preset Save/Load for Samsung Galaxy S22 calibration workflow
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Calibration Presets")
preset_name = st.sidebar.text_input("Preset Name", value="s22_yellow_tape")
if st.sidebar.button("💾 Save Calibration Preset"):
    save_target = Path("configs") / f"{preset_name.strip()}.yaml"
    save_config(st.session_state.config, save_target)
    st.sidebar.success(f"Saved: {save_target}")

# Telemetry logging toggle
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Telemetry Logging")
enable_log = st.sidebar.checkbox("Log to CSV", value=st.session_state.logging_enabled)
if enable_log != st.session_state.logging_enabled:
    st.session_state.logging_enabled = enable_log
    if enable_log:
        logger = TelemetryLogger(prefix="run")
        st.session_state.telemetry_logger = logger
        st.session_state.pipeline.telemetry_logger = logger
        st.sidebar.info(f"Logging active: `{logger.log_path.name}`")
    else:
        if st.session_state.telemetry_logger is not None:
            st.session_state.telemetry_logger.close()
            st.session_state.telemetry_logger = None
            st.session_state.pipeline.telemetry_logger = None

# ----------------- MAIN UI VIEW -----------------
st.title("🚗 Autonomous Path Following Demonstration")
st.markdown(
    '<div class="notice-banner">⚠️ <b>SIMULATION / LAPTOP PROTOTYPE — NO HARDWARE CONTROL</b><br>'
    'All motor PWM and steering outputs are software-in-the-loop proposals for Review-I validation. '
    'No physical Raspberry Pi, GPIO, or motor drivers are commanded.</div>',
    unsafe_allow_html=True,
)

# Playback controls
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 3])

if input_mode != "Synthetic Demo" or (input_mode == "Synthetic Demo" and synthetic_submode == "Streaming Sequence"):
    if st.session_state.is_playing:
        if ctrl_col1.button("⏸️ Pause", width="stretch"):
            st.session_state.is_playing = False
            st.rerun()
    else:
        if ctrl_col1.button("▶️ Play / Start", width="stretch"):
            st.session_state.is_playing = True
            st.rerun()

    if ctrl_col2.button("⏭️ Step Frame", width="stretch"):
        st.session_state.is_playing = False
        st.session_state.step_frame_requested = True
        st.rerun()

    if ctrl_col3.button("🔄 Reset Source", width="stretch"):
        if st.session_state.active_source is not None:
            st.session_state.active_source.reset()
        st.session_state.pipeline.reset()
        st.session_state.history.clear()
        st.rerun()

# Source management helper
def get_or_create_source() -> Optional[BaseFrameSource]:
    current_source = st.session_state.active_source

    if input_mode == "Synthetic Demo":
        if synthetic_submode == "Static Test Case":
            if not isinstance(current_source, SyntheticCaseSource):
                if current_source: current_source.release()
                current_source = SyntheticCaseSource(case_name=selected_case)
            else:
                current_source.set_case(selected_case)
        else:
            if not isinstance(current_source, SyntheticStreamSource):
                if current_source: current_source.release()
                current_source = SyntheticStreamSource(num_frames=150, loop=True)

    elif input_mode == "Video File":
        if video_path_to_use is None or not video_path_to_use.exists():
            return None
        if not isinstance(current_source, VideoFileSource) or current_source.file_path != video_path_to_use:
            if current_source: current_source.release()
            current_source = VideoFileSource(file_path=video_path_to_use, loop=True)

    elif input_mode == "Webcam":
        if not isinstance(current_source, WebcamSource) or current_source.device_index != webcam_idx:
            if current_source: current_source.release()
            current_source = WebcamSource(device_index=webcam_idx)

    st.session_state.active_source = current_source
    return current_source


source = get_or_create_source()

# Containers for dynamic updating
views_container = st.container()
metrics_container = st.container()
charts_container = st.container()

def render_pipeline_output(result: PipelineResult, status_note: str) -> None:
    # 1. Image Views
    with views_container:
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            st.subheader("1. Annotated Road View")
            # Convert BGR to RGB for Streamlit display
            rgb_annotated = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_annotated, width="stretch")

        with v_col2:
            st.subheader("2. Segmentation Mask")
            st.image(result.mask, width="stretch", clamp=True)

        with v_col3:
            st.subheader("3. Raw Camera Frame")
            rgb_raw = cv2.cvtColor(result.original_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_raw, width="stretch")

    # 2. Telemetry Cards
    with metrics_container:
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

        # State styling
        state_cls = {
            "STRAIGHT": "state-straight",
            "LEFT": "state-left",
            "RIGHT": "state-right",
            "PATH_LOST": "state-lost",
            "STOP": "state-lost",
        }.get(result.decision.state, "state-lost")

        with m_col1:
            st.markdown(
                f'<div class="metric-card"><div>Controller State</div>'
                f'<div class="{state_cls}">{result.decision.state}</div>'
                f'<small>{"Valid Path" if result.perception.path_valid else "Path Lost"}</small></div>',
                unsafe_allow_html=True,
            )

        with m_col2:
            err = result.perception.normalized_error
            st.markdown(
                f'<div class="metric-card"><div>Lateral Error</div>'
                f'<div style="font-size:24px; font-weight:bold;">{err:+.3f}</div>'
                f'<small>Range: [-1.0, +1.0]</small></div>',
                unsafe_allow_html=True,
            )

        with m_col3:
            st.markdown(
                f'<div class="metric-card"><div>Left PWM Proposal</div>'
                f'<div style="font-size:24px; font-weight:bold; color:#81d4fa;">{result.decision.left_pwm}</div>'
                f'<small>Motor 0-255</small></div>',
                unsafe_allow_html=True,
            )

        with m_col4:
            st.markdown(
                f'<div class="metric-card"><div>Right PWM Proposal</div>'
                f'<div style="font-size:24px; font-weight:bold; color:#ffcc80;">{result.decision.right_pwm}</div>'
                f'<small>Motor 0-255</small></div>',
                unsafe_allow_html=True,
            )

        with m_col5:
            st.markdown(
                f'<div class="metric-card"><div>Processing Latency</div>'
                f'<div style="font-size:24px; font-weight:bold;">{result.processing_ms:.1f} ms</div>'
                f'<small>Measured Monotonic</small></div>',
                unsafe_allow_html=True,
            )

        with m_col6:
            st.markdown(
                f'<div class="metric-card"><div>Frame Rate</div>'
                f'<div style="font-size:24px; font-weight:bold; color:#b39ddb;">{result.fps:.1f} FPS</div>'
                f'<small>Actual Laptop Rate</small></div>',
                unsafe_allow_html=True,
            )

        st.caption(f"Source: **{status_note}** | Reason: *{result.decision.reason}*")

    # Record into rolling chart history
    st.session_state.history.append({
        "frame": result.frame_index,
        "error": result.perception.normalized_error,
        "left_pwm": result.decision.left_pwm,
        "right_pwm": result.decision.right_pwm,
    })

    # 3. Rolling Charts
    with charts_container:
        st.markdown("### 📈 Live Telemetry History")
        chart_col1, chart_col2 = st.columns(2)
        if len(st.session_state.history) > 1:
            df_hist = pd.DataFrame(st.session_state.history).set_index("frame")
            with chart_col1:
                st.line_chart(df_hist[["error"]], height=220)
                st.caption("Normalized Lateral Error (0 = optical center, negative = left, positive = right)")
            with chart_col2:
                st.line_chart(df_hist[["left_pwm", "right_pwm"]], height=220)
                st.caption("Simulated Left/Right Differential PWM Proposals (0-255)")


# Frame Execution Logic
if source is None or not source.is_opened:
    st.warning("⚠️ Selected source is currently unavailable or unconfigured.")
    if input_mode == "Webcam":
        st.error(
            f"Could not open webcam index {webcam_idx}. "
            "Please check privacy settings, verify the device index, or switch to 'Synthetic Demo'."
        )
    elif input_mode == "Video File":
        st.info("Please select or upload a video file to begin processing.")
else:
    # Single-case mode runs on every rerun (interactive slider tweaking)
    if input_mode == "Synthetic Demo" and synthetic_submode == "Static Test Case":
        ok, frame, msg = source.read_frame()
        if ok and frame is not None:
            res = st.session_state.pipeline.process_frame(frame, source_name=f"syn_{selected_case}")
            render_pipeline_output(res, msg)

    # Dynamic playback modes (Video / Stream / Webcam)
    elif st.session_state.is_playing:
        # Stream frame loop with placeholder
        live_placeholder = st.empty()
        
        # Process a sequence chunk smoothly
        for _ in range(30):
            if not st.session_state.is_playing:
                break
            ok, frame, msg = source.read_frame()
            if not ok or frame is None:
                st.session_state.is_playing = False
                st.warning(f"Playback ended: {msg}")
                break

            res = st.session_state.pipeline.process_frame(frame, source_name=input_mode)
            render_pipeline_output(res, msg)
            time.sleep(0.02)  # Yield approx 30-50 FPS
        
        if st.session_state.is_playing:
            st.rerun()

    else:
        # Paused or step mode: process and show current/next single frame
        ok, frame, msg = source.read_frame()
        if ok and frame is not None:
            res = st.session_state.pipeline.process_frame(frame, source_name=input_mode)
            render_pipeline_output(res, msg)
        else:
            st.info(f"Stream paused or at end: {msg}")
