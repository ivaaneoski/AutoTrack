import sys
from pathlib import Path

# Ensure repository root is on Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from pathdemo.config import PerceptionConfig
from pathdemo.perception import process_path_frame
from pathdemo.synthetic import generate_synthetic_case


def main():
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PerceptionConfig()

    cases = [
        "straight_centered",
        "straight_left",
        "straight_right",
        "curve_left",
        "curve_right",
        "noisy_centered",
        "missing_path",
    ]

    for case in cases:
        syn = generate_synthetic_case(case, width=640, height=480)
        res = process_path_frame(syn.frame, cfg)

        cv2.imwrite(str(out_dir / f"{case}_raw.png"), syn.frame)
        cv2.imwrite(str(out_dir / f"{case}_annotated.png"), res.annotated_frame)
        cv2.imwrite(str(out_dir / f"{case}_mask.png"), res.mask)
        cv2.imwrite(str(out_dir / f"{case}_roi_mask.png"), res.roi_mask)

        print(f"Saved figures for case '{case}': valid={res.path_valid}, error={res.normalized_error:+.3f}")


if __name__ == "__main__":
    main()
