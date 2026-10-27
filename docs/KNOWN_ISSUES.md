# Known Issues and Limitations

This document tracks known constraints, hardware boundaries, and edge cases for the Review-I prototype.

## Fundamental MVP Boundaries
1. **Simulation-Only Motor Output:**
   - Motor outputs (`left_pwm`, `right_pwm`) are simulated proposals bounded to `[0, 255]`.
   - No hardware control (Raspberry Pi GPIO, L298N motor drivers, Arduino PWM) is connected in this prototype.
   - Laptop CPU processing does not guarantee real-time embedded timing on a physical Raspberry Pi.

2. **Classic CV Baseline Lighting Sensitivity:**
   - The HSV color detector relies on consistent illumination contrast between the target path (e.g. yellow tape) and the surface floor/paper.
   - Glare, extreme shadows, or floor colors that fall into the yellow HSV spectrum require adjusting threshold sliders in the dashboard.
   - Dynamic auto-white balance on phone cameras (e.g., Samsung Galaxy S22) can shift hue values across clips. Slider calibration is provided to compensate.

3. **Single Baseline Camera Source:**
   - The system assumes a forward-facing camera with the road in the lower Region of Interest (ROI).
   - If the vehicle completely leaves the track, the detector issues a `PATH_LOST` flag and commands `STOP` (0 PWM) as a safety mechanism. Recovery/search maneuvers are reserved for future work.

4. **Phone Video Codecs (Samsung Galaxy S22):**
   - Standard 1080p MP4 recordings encoded in H.264 (AVC) work out-of-the-box with Windows OpenCV Media Foundation / DirectShow.
   - If Samsung's "High Efficiency Video" (HEVC / H.265) camera option is enabled without system HEVC codecs, OpenCV `VideoCapture` may fail to decode frames.
   - **Resolution:** In Galaxy S22 Camera Settings -> Advanced Video Options, ensure "High efficiency videos" is switched to standard H.264 MP4.

5. **Track Bifurcations & Crossings:**
   - At a 90-degree crossing or Y-split fork, the single largest connected contour combines both branches, placing the centroid at the junction geometric average.
   - Advanced lane selection or multi-band lookahead can be integrated in Review-II.
