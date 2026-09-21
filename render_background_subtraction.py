"""Render registered-background subtraction diagnostics for chosen timestamps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from perception.core import FrameAnalyzer, _scene, read_video_frame, video_frame_timestamps
from perception.workspace import MapWorkspace


HERE = Path(__file__).resolve().parent


def remove_fixed_ui(mask: np.ndarray, valid_gameplay_mask: np.ndarray) -> np.ndarray:
    """Remove capture-local fixed HUD regions from a scene-space mask."""
    if mask.shape != valid_gameplay_mask.shape:
        raise ValueError(
            f"Residual/UI mask shape mismatch: {mask.shape} != {valid_gameplay_mask.shape}"
        )
    return mask & (valid_gameplay_mask > 0)


def _label(image: np.ndarray, title: str, detail: str = "") -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(canvas, detail, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def _expected_view(
    analyzer: FrameAnalyzer,
    coverage_canvas: np.ndarray,
    scene: np.ndarray,
    shift_x: float,
    shift_y: float,
):
    profile = analyzer.profile
    x0 = int(round(profile.canvas_origin_x - shift_x))
    y0 = int(round(profile.canvas_origin_y - shift_y))
    height, width = scene.shape[:2]
    expected = np.zeros_like(scene)
    stable = np.zeros((height, width), np.uint8)
    valid = np.zeros((height, width), np.uint8)
    coverage = np.zeros((height, width), np.uint8)
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1 = min(analyzer.background.shape[1], x0 + width)
    src_y1 = min(analyzer.background.shape[0], y0 + height)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        dst_x1 = dst_x0 + src_x1 - src_x0
        dst_y1 = dst_y0 + src_y1 - src_y0
        expected[dst_y0:dst_y1, dst_x0:dst_x1] = analyzer.background[src_y0:src_y1, src_x0:src_x1]
        stable[dst_y0:dst_y1, dst_x0:dst_x1] = analyzer.stable[src_y0:src_y1, src_x0:src_x1]
        coverage[dst_y0:dst_y1, dst_x0:dst_x1] = coverage_canvas[src_y0:src_y1, src_x0:src_x1]
        valid[dst_y0:dst_y1, dst_x0:dst_x1] = 255
    return expected, stable, coverage, valid


def render_sample(
    analyzer: FrameAnalyzer,
    coverage_canvas: np.ndarray,
    frame: np.ndarray,
    timestamp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene = _scene(frame, analyzer.profile)
    registration = analyzer.registrar.register(scene)
    if not registration.ok or registration.shift_x is None or registration.shift_y is None:
        raise RuntimeError(f"Registration failed at {timestamp:.3f}s")
    expected, stable, coverage, valid = _expected_view(
        analyzer, coverage_canvas, scene, registration.shift_x, registration.shift_y
    )
    raw_difference = np.max(cv2.absdiff(scene, expected), axis=2)
    residual, _ = analyzer._registered_residual(scene, registration)
    covered = coverage > 0
    threshold = int(analyzer.profile.thresholds.get("background_residual", 34))
    all_covered_residual = ((raw_difference >= threshold) & covered).astype(np.uint8) * 255
    all_covered_residual = cv2.morphologyEx(
        all_covered_residual, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    unstable_residual = (all_covered_residual > 0) & (stable < 128)
    unstable_residual = remove_fixed_ui(
        unstable_residual, analyzer.registrar.valid_mask
    )

    # Expected background: show the map reconstruction normally where covered,
    # and darken pixels that are not trusted by the conservative stable mask.
    expected_display = (expected.astype(np.float32) * 0.28).astype(np.uint8)
    expected_display[stable >= 128] = expected[stable >= 128]
    expected_display[valid == 0] = (28, 28, 28)

    # Overlay: stable comparison area is mildly desaturated; thresholded
    # residual pixels are magenta. Everything else is explicitly unknown/dim.
    gray = cv2.cvtColor(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    overlay = (gray.astype(np.float32) * 0.32).astype(np.uint8)
    overlay[stable >= 128] = (scene[stable >= 128].astype(np.float32) * 0.72).astype(np.uint8)
    overlay[residual > 0] = (255, 0, 255)

    # Experimental all-covered overlay. Stable residual stays magenta; newly
    # admitted unstable residual is yellow. Covered non-residual pixels remain
    # visible, while genuinely uncovered panorama gaps stay gray.
    all_overlay = (gray.astype(np.float32) * 0.28).astype(np.uint8)
    all_overlay[covered] = (scene[covered].astype(np.float32) * 0.62).astype(np.uint8)
    all_overlay[residual > 0] = (255, 0, 255)
    all_overlay[unstable_residual] = (0, 255, 255)

    stable_fraction = np.count_nonzero(stable >= 128) / stable.size
    residual_fraction = np.count_nonzero(residual) / max(1, np.count_nonzero(stable >= 128))
    covered_fraction = np.count_nonzero(covered) / covered.size
    all_residual_fraction = np.count_nonzero(all_covered_residual) / max(1, np.count_nonzero(covered))
    registration_text = f"shift=({registration.shift_x:.1f},{registration.shift_y:.1f})  inliers={registration.inliers}"
    panels = [
        _label(scene, f"Captured scene  t={timestamp:.3f}s", registration_text),
        _label(expected_display, "Expected registered map", f"bright=stable; dim=unstable; gray=uncovered  stable={stable_fraction:.1%}"),
        _label(overlay, "Stable-only subtraction", f"magenta=stable residual  residual/stable={residual_fraction:.1%}"),
        _label(
            all_overlay,
            "All-covered subtraction (experimental)",
            f"magenta=stable; yellow=unstable residual  covered={covered_fraction:.1%} residual={all_residual_fraction:.1%}",
        ),
    ]
    target_size = (480, 229)
    comparison = np.hstack([cv2.resize(panel, target_size, interpolation=cv2.INTER_AREA) for panel in panels])

    # Preserve captured RGB only for thresholded residuals in covered but
    # unstable map regions. Fixed HUD rectangles are always blacked out before
    # this image is handed to a detector.
    unstable_cutout = np.zeros_like(scene)
    unstable_cutout[unstable_residual] = scene[unstable_residual]
    binary_mask = unstable_residual.astype(np.uint8) * 255
    return comparison, unstable_cutout, binary_mask


def main() -> int:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--map", default=os.environ.get("SVD_MAP", "rednose3"))
    selected, _ = selector.parse_known_args()
    workspace = MapWorkspace.load(HERE, selected.map)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=workspace.map_id)
    parser.add_argument("--video", type=Path, default=workspace.video)
    parser.add_argument("--profile", type=Path, default=workspace.path("profile"))
    parser.add_argument(
        "--coverage",
        type=Path,
        default=workspace.capture_dir / "background_reconstruction" / "map_aligned_coverage.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace.path("proof") / "background_subtraction_with_unstable.png",
    )
    parser.add_argument(
        "--cutout-output",
        type=Path,
        default=workspace.path("proof") / "unstable_residual_cutouts.png",
    )
    parser.add_argument(
        "--cutout-dir",
        type=Path,
        default=workspace.path("proof") / "unstable_residual_cases",
    )
    parser.add_argument(
        "--timestamps", type=float, nargs="+",
        default=workspace.config.get("diagnostics", {}).get("background_subtraction_timestamps"),
    )
    args = parser.parse_args()
    if not args.timestamps:
        raise ValueError(
            f"Map {workspace.map_id!r} has no background-subtraction timestamps configured; "
            "pass --timestamps"
        )

    timestamps = video_frame_timestamps(args.video)
    analyzer = FrameAnalyzer(args.profile)
    coverage_canvas = cv2.imread(str(args.coverage), cv2.IMREAD_GRAYSCALE)
    if coverage_canvas is None:
        raise FileNotFoundError(f"Could not read coverage map: {args.coverage}")
    if coverage_canvas.shape != analyzer.stable.shape:
        raise ValueError(
            f"Coverage shape {coverage_canvas.shape} does not match map shape {analyzer.stable.shape}"
        )
    rows = []
    cutout_rows = []
    args.cutout_dir.mkdir(parents=True, exist_ok=True)
    for requested in args.timestamps:
        frame_index = min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - requested))
        timestamp = timestamps[frame_index]
        frame = read_video_frame(args.video, frame_index=frame_index)
        comparison, cutout, binary_mask = render_sample(
            analyzer, coverage_canvas, frame, timestamp
        )
        rows.append(comparison)
        surviving_fraction = np.count_nonzero(binary_mask) / binary_mask.size
        cutout_panel = _label(
            cutout,
            f"Unstable residual pixels only  t={timestamp:.3f}s",
            f"original RGB survives; black=removed  surviving={surviving_fraction:.1%}",
        )
        cutout_rows.append(
            cv2.resize(cutout_panel, (960, 458), interpolation=cv2.INTER_AREA)
        )
        stem = f"unstable_residual_{timestamp:010.3f}"
        if not cv2.imwrite(str(args.cutout_dir / f"{stem}.png"), cutout):
            raise RuntimeError(f"Could not write cutout for {timestamp:.3f}s")
        if not cv2.imwrite(str(args.cutout_dir / f"{stem}_mask.png"), binary_mask):
            raise RuntimeError(f"Could not write mask for {timestamp:.3f}s")
        analyzer.reset()
    output = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise RuntimeError(f"Could not write {args.output}")
    cutout_sheet = np.vstack(cutout_rows)
    if not cv2.imwrite(str(args.cutout_output), cutout_sheet):
        raise RuntimeError(f"Could not write {args.cutout_output}")
    print(args.output)
    print(args.cutout_output)
    print(args.cutout_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
