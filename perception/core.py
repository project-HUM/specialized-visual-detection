"""Capture-local visual perception for the pirate-ship recording.

This is a conservative baseline, not a general MapleStory recognizer.  It
registers the map layer, matches session-specific character and monster
templates, and segments the distinctive black/red A-skill residual.  Every
recognizer is allowed to abstain.
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
from .tracking import MonsterTracker
from .specialized_detector import SpecializedMonsterDetector, TemplateMonsterDetector


PROFILE_VERSION = 1
FIXED_UI_RECTS = (
    (0, 0, 1920, 30),
    (0, 0, 380, 350),
    (1510, 0, 1920, 245),
    (0, 330, 110, 790),
    (0, 820, 1620, 1080),
    (1620, 920, 1920, 1080),
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def read_video_frame(video: Path, *, frame_index: int | None = None, timestamp: float | None = None) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    if frame_index is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    elif timestamp is not None:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        target = f"frame {frame_index}" if frame_index is not None else f"time {timestamp}"
        raise RuntimeError(f"Could not decode {target} from {video}")
    return frame


def probe_frame_timestamps(video: Path) -> list[float]:
    """Return presentation timestamps; never derive time from nominal FPS."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0", str(video),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    timestamps: list[float] = []
    for line in result.stdout.splitlines():
        value = line.strip().split(",", 1)[0]
        if value:
            timestamps.append(float(value))
    if not timestamps:
        raise RuntimeError(f"ffprobe returned no frame timestamps for {video}")
    return timestamps


def video_frame_timestamps(video: Path) -> list[float]:
    """Prefer the capture's synchronized per-frame PTS cache when available."""
    positions = video.parent / "map_output" / "positions.csv"
    if positions.is_file():
        by_frame: dict[int, float] = {}
        for source in (positions, positions.parent / "missing_frames.csv"):
            if not source.is_file():
                continue
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    by_frame[int(row["frame"])] = float(row["time_s"])
        if by_frame:
            last = max(by_frame)
            if len(by_frame) == last + 1:
                # The background reconstruction intentionally stops before the
                # final map-changing minute, so this cache is also the maximum
                # valid interval for the capture-local profile.
                return [by_frame[index] for index in range(last + 1)]
    return probe_frame_timestamps(video)


@dataclass
class SessionProfile:
    version: int
    profile_id: str
    source_width: int
    source_height: int
    scene_top: int
    scene_bottom: int
    scale: float
    reference_frame: int
    reference_time_s: float
    reference_scene: str
    background_image: str
    stable_mask: str
    variability_image: str
    canvas_origin_x: int
    canvas_origin_y: int
    character_templates: dict[str, list[str]] = field(default_factory=dict)
    monster_templates: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    calibration_quality: dict[str, Any] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    profile_path: Path | None = field(default=None, repr=False, compare=False)

    @property
    def base_dir(self) -> Path:
        if self.profile_path is None:
            raise RuntimeError("Profile has not been saved or loaded")
        return self.profile_path.parent

    def asset(self, relative: str) -> Path:
        return (self.base_dir / relative).resolve()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload.pop("profile_path", None)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.profile_path = path.resolve()

    @classmethod
    def load(cls, path: Path | str) -> "SessionProfile":
        profile_path = Path(path).resolve()
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != PROFILE_VERSION:
            raise ValueError(f"Unsupported SessionProfile version: {payload.get('version')}")
        profile = cls(**payload)
        profile.profile_path = profile_path
        return profile


@dataclass(frozen=True)
class RegistrationResult:
    ok: bool
    shift_x: float | None
    shift_y: float | None
    good_matches: int
    inliers: int
    confidence: float


class MapRegistrar:
    def __init__(self, profile: SessionProfile):
        self.profile = profile
        self.reference = _read_image(profile.asset(profile.reference_scene))
        self.valid_mask = self._ui_mask(self.reference.shape[:2])
        self.sift = cv2.SIFT_create(nfeatures=8_000, contrastThreshold=0.018)
        self.ref_keypoints, self.ref_descriptors = self._describe(self.reference)
        if self.ref_descriptors is None or len(self.ref_keypoints) < 20:
            raise RuntimeError("Reference scene does not contain enough registration features")

    def _ui_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        source = np.full((self.profile.scene_bottom - self.profile.scene_top, self.profile.source_width), 255, np.uint8)
        for left, top, right, bottom in FIXED_UI_RECTS:
            local_top = max(0, min(source.shape[0], top - self.profile.scene_top))
            local_bottom = max(0, min(source.shape[0], bottom - self.profile.scene_top))
            left = max(0, min(source.shape[1], left))
            right = max(0, min(source.shape[1], right))
            if right > left and local_bottom > local_top:
                source[local_top:local_bottom, left:right] = 0
        return cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST)

    def _describe(self, scene: np.ndarray):
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
        rows = np.indices(self.valid_mask.shape)[0]
        useful = (self.valid_mask > 0) & ((hsv[:, :, 1] >= 42) | (rows >= int(scene.shape[0] * 0.5)))
        return self.sift.detectAndCompute(gray, useful.astype(np.uint8) * 255)

    def register(self, scene: np.ndarray) -> RegistrationResult:
        keypoints, descriptors = self._describe(scene)
        if descriptors is None or len(keypoints) < 12:
            return RegistrationResult(False, None, None, 0, 0, 0.0)
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(self.ref_descriptors, descriptors, k=2)
        good = [first for first, second in pairs if first.distance < 0.70 * second.distance]
        if len(good) < 12:
            return RegistrationResult(False, None, None, len(good), 0, min(0.3, len(good) / 40.0))
        vectors = np.asarray([
            (keypoints[m.trainIdx].pt[0] - self.ref_keypoints[m.queryIdx].pt[0],
             keypoints[m.trainIdx].pt[1] - self.ref_keypoints[m.queryIdx].pt[1])
            for m in good
        ], dtype=np.float32)
        bins = np.rint(vectors / 2.0).astype(np.int32)
        unique_bins, counts = np.unique(bins, axis=0, return_counts=True)
        peak = unique_bins[int(np.argmax(counts))] * 2.0
        residual = np.linalg.norm(vectors - peak, axis=1)
        inlier_vectors = vectors[residual <= 3.0]
        inliers = len(inlier_vectors)
        if inliers < 10:
            return RegistrationResult(False, None, None, len(good), inliers, min(0.45, inliers / 25.0))
        shift_x, shift_y = np.median(inlier_vectors, axis=0)
        confidence = min(1.0, 0.45 + inliers / 80.0) * min(1.0, inliers / max(1.0, len(good) * 0.35))
        return RegistrationResult(True, float(shift_x), float(shift_y), len(good), inliers, float(confidence))


def _scene(frame: np.ndarray, profile: SessionProfile) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR color image")
    height, width = frame.shape[:2]
    if (width, height) != (profile.source_width, profile.source_height):
        raise ValueError(
            f"Resolution mismatch: expected {profile.source_width}x{profile.source_height}, got {width}x{height}"
        )
    crop = frame[profile.scene_top:profile.scene_bottom]
    return cv2.resize(crop, None, fx=profile.scale, fy=profile.scale, interpolation=cv2.INTER_AREA)


def _nms(points: list[tuple[float, float, float, int, int]], radius: float) -> list[tuple[float, float, float, int, int]]:
    kept: list[tuple[float, float, float, int, int]] = []
    for point in sorted(points, key=lambda p: p[2], reverse=True):
        if all(math.hypot(point[0] - other[0], point[1] - other[1]) >= radius for other in kept):
            kept.append(point)
    return kept


class FrameAnalyzer:
    """Stateful frame API.  State is incremental and never rereads the video."""

    def __init__(self, profile: SessionProfile | Path | str,
                 monster_detector: SpecializedMonsterDetector | None = None):
        self.profile = profile if isinstance(profile, SessionProfile) else SessionProfile.load(profile)
        self.registrar = MapRegistrar(self.profile)
        self.background = _read_image(self.profile.asset(self.profile.background_image))
        self.stable = _read_image(self.profile.asset(self.profile.stable_mask), cv2.IMREAD_GRAYSCALE)
        self.variability = _read_image(self.profile.asset(self.profile.variability_image), cv2.IMREAD_GRAYSCALE)
        self.character_templates = {
            facing: [_read_image(self.profile.asset(path), cv2.IMREAD_GRAYSCALE) for path in paths]
            for facing, paths in self.profile.character_templates.items()
        }
        self.monster_templates = [
            _read_image(self.profile.asset(path), cv2.IMREAD_GRAYSCALE) for path in self.profile.monster_templates
        ]
        self.monster_detector = monster_detector or TemplateMonsterDetector(
            self.monster_templates, threshold=float(self.profile.thresholds.get("monster_match", 0.42))
        )
        self.previous_gray: np.ndarray | None = None
        self.previous_wave_centroid: tuple[float, float] | None = None
        self.previous_timestamp: float | None = None
        self.previous_character_center: list[float] | None = None
        self.previous_character_timestamp: float | None = None
        self.frame_counter = 0
        self.monster_tracker = MonsterTracker(
            min_hits=int(self.profile.thresholds.get("monster_min_hits", 2)),
            max_gap_s=float(self.profile.thresholds.get("monster_track_gap_s", 0.55)),
            base_match_distance=float(self.profile.thresholds.get("monster_track_distance", 28.0)),
            supported_occlusion_s=float(self.profile.thresholds.get("monster_supported_occlusion_s", 2.5)),
            dedup_iou=float(self.profile.thresholds.get("monster_dedup_iou", 0.88)),
            dedup_center_fraction=float(self.profile.thresholds.get("monster_dedup_center_fraction", 0.22)),
            dedup_size_similarity=float(self.profile.thresholds.get("monster_dedup_size_similarity", 0.72)),
        )

    def reset(self) -> None:
        self.previous_gray = None
        self.previous_wave_centroid = None
        self.previous_timestamp = None
        self.previous_character_center = None
        self.previous_character_timestamp = None
        self.frame_counter = 0
        self.monster_tracker.reset()

    def _registered_residual(self, scene: np.ndarray, registration: RegistrationResult):
        if not registration.ok or registration.shift_x is None or registration.shift_y is None:
            empty = np.zeros(scene.shape[:2], np.uint8)
            return empty, empty
        x0 = int(round(self.profile.canvas_origin_x - registration.shift_x))
        y0 = int(round(self.profile.canvas_origin_y - registration.shift_y))
        height, width = scene.shape[:2]
        expected = np.zeros_like(scene)
        stable = np.zeros((height, width), np.uint8)
        src_x0, src_y0 = max(0, x0), max(0, y0)
        src_x1, src_y1 = min(self.background.shape[1], x0 + width), min(self.background.shape[0], y0 + height)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return stable, stable
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        dst_x1, dst_y1 = dst_x0 + src_x1 - src_x0, dst_y0 + src_y1 - src_y0
        expected[dst_y0:dst_y1, dst_x0:dst_x1] = self.background[src_y0:src_y1, src_x0:src_x1]
        stable[dst_y0:dst_y1, dst_x0:dst_x1] = self.stable[src_y0:src_y1, src_x0:src_x1]
        difference = np.max(cv2.absdiff(scene, expected), axis=2)
        threshold = int(self.profile.thresholds.get("background_residual", 34))
        residual = ((difference >= threshold) & (stable >= 128)).astype(np.uint8) * 255
        residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return residual, stable

    def _match_character(self, scene: np.ndarray) -> tuple[dict[str, Any], list[str]]:
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 55, 140)
        scores: dict[str, tuple[float, tuple[int, int], tuple[int, int]]] = {}
        for facing, templates in self.character_templates.items():
            best = (-1.0, (0, 0), (0, 0))
            for template in templates:
                if template.shape[0] > edge.shape[0] or template.shape[1] > edge.shape[1]:
                    continue
                result = cv2.matchTemplate(edge, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(result)
                if score > best[0]:
                    best = (float(score), location, (template.shape[1], template.shape[0]))
            scores[facing] = best
        ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
        flags: list[str] = []
        if not ranked or ranked[0][1][0] < self.profile.thresholds.get("character_match", 0.38):
            flags.append("character_not_visible")
            return {"center": None, "facing": "unknown", "confidence": 0.0, "provenance": "visual"}, flags
        facing, (top_score, location, size) = ranked[0]
        second_score = ranked[1][1][0] if len(ranked) > 1 else -1.0
        margin = top_score - second_score
        center_scene = (location[0] + size[0] / 2.0, location[1] + size[1] / 2.0)
        center_source = [
            center_scene[0] / self.profile.scale,
            center_scene[1] / self.profile.scale + self.profile.scene_top,
        ]
        min_margin = self.profile.thresholds.get("facing_margin", 0.035)
        visible_facing = facing if margin >= min_margin else "unknown"
        confidence = max(0.0, min(1.0, (top_score - 0.25) / 0.35))
        if visible_facing == "unknown":
            confidence *= 0.55
            flags.append("facing_ambiguous")
        return {
            "center": [round(float(center_source[0]), 2), round(float(center_source[1]), 2)],
            "facing": visible_facing,
            "confidence": round(float(confidence), 4),
            "provenance": "visual",
            "scores": {name: round(float(value[0]), 4) for name, value in scores.items()},
        }, flags

    def _detect_monsters(self, scene: np.ndarray, timestamp: float | None) -> list[dict[str, Any]]:
        """Normalize any detector backend from scene pixels to source pixels."""
        monsters: list[dict[str, Any]] = []
        for detection in self.monster_detector.detect(scene, timestamp):
            x1, y1, x2, y2 = detection.bbox
            box = [x1/self.profile.scale, y1/self.profile.scale+self.profile.scene_top,
                   x2/self.profile.scale, y2/self.profile.scale+self.profile.scene_top]
            gx, gy = detection.ground_position
            monsters.append({"box":[round(float(v),2) for v in box],
                             "score":float(detection.detector_confidence),
                             "ground_position":[round(gx/self.profile.scale,2),round(gy/self.profile.scale+self.profile.scene_top,2)],
                             "backend":detection.backend,"provenance":f"visual_{detection.backend}",
                             "metadata":detection.metadata or {}})
        return monsters

    def _detect_wave(
        self,
        scene: np.ndarray,
        residual: np.ndarray,
        character_center: Sequence[float] | None,
        timestamp: float | None,
    ) -> tuple[dict[str, Any], list[str]]:
        flags: list[str] = []
        if character_center is None:
            return {"state": "unknown", "direction": "unknown", "confidence": 0.0, "provenance": "visual"}, ["wave_character_missing"]
        cx = character_center[0] * self.profile.scale
        cy = (character_center[1] - self.profile.scene_top) * self.profile.scale
        hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
        red = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 172)) & (hsv[:, :, 1] >= 115) & (hsv[:, :, 2] >= 45))
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        temporal = np.zeros_like(gray, dtype=bool)
        flow_dx: float | None = None
        continuous = (
            self.previous_gray is not None
            and self.previous_gray.shape == gray.shape
            and (timestamp is None or self.previous_timestamp is None or 0.0 <= timestamp - self.previous_timestamp <= 0.25)
        )
        if continuous:
            temporal = cv2.absdiff(gray, self.previous_gray) >= 24
            flow = cv2.calcOpticalFlowFarneback(self.previous_gray, gray, None, 0.5, 3, 15, 2, 5, 1.1, 0)
        else:
            flow = None
            self.previous_wave_centroid = None
        half_w = int(460 * self.profile.scale)
        half_h = int(190 * self.profile.scale)
        x0, x1 = max(0, int(cx - half_w)), min(scene.shape[1], int(cx + half_w))
        y0, y1 = max(0, int(cy - half_h)), min(scene.shape[0], int(cy + half_h))
        roi = np.zeros_like(gray, np.uint8)
        roi[y0:y1, x0:x1] = 255
        seed = red & ((residual > 0) | temporal) & (roi > 0)
        seed = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        ys, xs = np.where(seed > 0)
        area = int(len(xs))
        minimum = int(self.profile.thresholds.get("wave_red_area", 24))
        if area < minimum:
            self.previous_wave_centroid = None
            return {"state": "inactive", "direction": "inactive", "confidence": round(min(0.85, 0.4 + (minimum - area) / max(1, minimum) * 0.35), 4), "provenance": "visual"}, flags
        centroid = (float(np.median(xs)), float(np.median(ys)))
        position_dx = centroid[0] - cx
        position_direction = "unknown"
        if abs(position_dx) >= self.profile.thresholds.get("wave_position_margin", 18.0):
            position_direction = "right" if position_dx > 0 else "left"
        if flow is not None:
            values = flow[:, :, 0][seed > 0]
            values = values[np.isfinite(values)]
            if len(values):
                flow_dx = float(np.median(values))
        flow_direction = "unknown"
        if flow_dx is not None and abs(flow_dx) >= self.profile.thresholds.get("wave_flow_margin", 0.35):
            flow_direction = "right" if flow_dx > 0 else "left"
        track_direction = "unknown"
        if self.previous_wave_centroid is not None:
            delta = centroid[0] - self.previous_wave_centroid[0]
            if abs(delta) >= 1.5:
                track_direction = "right" if delta > 0 else "left"
        votes = [value for value in (position_direction, flow_direction, track_direction) if value != "unknown"]
        direction = "unknown"
        if votes:
            left_votes, right_votes = votes.count("left"), votes.count("right")
            if left_votes and right_votes and max(left_votes, right_votes) == 1:
                flags.append("wave_direction_disagreement")
            else:
                direction = "left" if left_votes > right_votes else "right"
        if position_direction == "unknown" and track_direction == "unknown":
            flags.append("wave_symmetric_onset")
        confidence = min(0.98, 0.45 + area / max(minimum * 8.0, 1.0))
        if direction == "unknown":
            confidence *= 0.55
        self.previous_wave_centroid = centroid
        return {
            "state": "active",
            "direction": direction,
            "confidence": round(float(confidence), 4),
            "provenance": "visual",
            "evidence": {
                "red_residual_area": area,
                "position_dx": round(float(position_dx / self.profile.scale), 2),
                "flow_dx": None if flow_dx is None else round(float(flow_dx / self.profile.scale), 3),
                "position_direction": position_direction,
                "flow_direction": flow_direction,
                "track_direction": track_direction,
            },
        }, flags

    @staticmethod
    def _fuse_keys(character: dict[str, Any], wave: dict[str, Any], key_state: dict[str, Any] | None, flags: list[str]) -> None:
        if not key_state:
            return
        # A stale last movement key is not evidence of the currently visible
        # sprite facing.  Only an exclusive direction that is held now may be
        # fused; the timeline still exposes last direction for diagnostics.
        key_direction = key_state.get("direction") if key_state.get("direction_held") else None
        if key_direction in ("left", "right"):
            visual = character["facing"]
            if visual == "unknown":
                character["facing"] = key_direction
                character["provenance"] = "keys"
                character["confidence"] = max(character["confidence"], 0.55)
            elif visual == key_direction:
                character["provenance"] = "fused"
                character["confidence"] = min(1.0, character["confidence"] + 0.08)
            else:
                flags.append("character_direction_key_disagreement")
        a_age = key_state.get("a_age_s")
        a_recent = isinstance(a_age, (int, float)) and 0.0 <= float(a_age) <= 0.85
        if a_recent:
            if wave["state"] == "inactive":
                flags.append("a_key_without_visual_wave")
                wave["state"] = "unknown"
                wave["direction"] = "unknown"
                wave["provenance"] = "keys"
                wave["confidence"] = 0.3
            elif wave["state"] == "active":
                wave["provenance"] = "fused"
                if key_direction in ("left", "right"):
                    if wave["direction"] == key_direction:
                        wave["confidence"] = min(1.0, wave["confidence"] + 0.08)
                    elif wave["direction"] not in ("unknown", "inactive"):
                        flags.append("wave_direction_key_disagreement")

    def analyze(self, image: np.ndarray, timestamp: float | None = None, key_state: dict[str, Any] | None = None) -> dict[str, Any]:
        scene = _scene(image, self.profile)
        registration = self.registrar.register(scene)
        flags: list[str] = []
        if not registration.ok:
            flags.append("registration_failed")
        elif registration.confidence < 0.7:
            flags.append("registration_low_confidence")
        residual, stable = self._registered_residual(scene, registration)
        character, character_flags = self._match_character(scene)
        flags.extend(character_flags)
        if character["center"] is None and self.previous_character_center is not None:
            age = None if timestamp is None or self.previous_character_timestamp is None else timestamp - self.previous_character_timestamp
            if age is None or 0.0 <= age <= 0.60:
                character["center"] = list(self.previous_character_center)
                character["confidence"] = 0.25
                character["facing"] = "unknown"
                character["provenance"] = "visual_track"
                flags.append("character_temporal_hold")
        raw_monsters = self._detect_monsters(scene, timestamp)
        camera_origin = ((registration.shift_x or 0.0) / self.profile.scale,
                         (registration.shift_y or 0.0) / self.profile.scale)
        monster_tracks = self.monster_tracker.update(
            raw_monsters,
            float(timestamp if timestamp is not None else self.frame_counter / 30.0),
            # Template boxes are in source-pixel coordinates while registration
            # runs on the half-resolution scene; convert the camera translation
            # before updating map-space tracks.
            camera_origin=camera_origin,
            exclusion_boxes=([[character["center"][0] - 45, character["center"][1] - 75,
                               character["center"][0] + 45, character["center"][1] + 75]]
                             if character["center"] is not None else ()),
            include_tentative=False,
        )
        monster_quality: list[str] = []
        for track in monster_tracks:
            if character["center"] is not None:
                track["relative_to_character"] = [
                    round(track["ground_position"][0] - character["center"][0], 2),
                    round(track["ground_position"][1] - character["center"][1], 2),
                ]
            else:
                track["relative_to_character"] = None
            monster_quality.extend(track.get("quality_flags", []))
        flags.extend(sorted(set(monster_quality)))
        visual_detections = []
        for item in self.monster_tracker.last_debug["retained_detections"]:
            box = [item["box"][0]-camera_origin[0], item["box"][1]-camera_origin[1],
                   item["box"][2]-camera_origin[0], item["box"][3]-camera_origin[1]]
            visual_detections.append({"box":[round(float(v),2) for v in box],
                                      "center":[round((box[0]+box[2])/2,2),round((box[1]+box[3])/2,2)],
                                      "ground_position":[round((box[0]+box[2])/2,2),round(box[3],2)],
                                      "confidence":round(float(item.get("score",0)),4),
                                      "provenance":item.get("provenance",f"visual_{item.get('backend','unknown')}")})
        wave, wave_flags = self._detect_wave(scene, residual, character["center"], timestamp)
        flags.extend(wave_flags)
        visual_only = {"character": copy.deepcopy(character), "a_wave": copy.deepcopy(wave)}
        self._fuse_keys(character, wave, key_state, flags)
        if not registration.ok:
            # Template evidence remains usable, but background-derived wave evidence does not.
            wave = {"state": "unknown", "direction": "unknown", "confidence": 0.0, "provenance": "visual"}
        estimated_count = self.monster_tracker.estimated_count()
        count_min, count_max = self.monster_tracker.count_bounds()
        observation = {
            "timestamp": timestamp,
            "frame_index": self.frame_counter,
            "character": character,
            "monsters": {
                # ``visual_detection_count`` is current retained detector evidence.
                # ``estimated_count`` is live confirmed persistent identities.
                # Bounds combine current independent visual evidence units with
                # temporal identity evidence; a merged group contributes one to
                # the lower bound and all credible members to the upper bound.
                "count": estimated_count,
                "visual_detection_count": len(self.monster_tracker.last_debug["retained_detections"]),
                "estimated_count": estimated_count,
                "count_min": count_min,
                "count_max": count_max,
                "centers": monster_tracks,
                "tracks": monster_tracks,
                "visual_detections": visual_detections,
                "confidence": round(float(np.mean([m["confidence"] for m in monster_tracks])) if monster_tracks else 0.0, 4),
                "provenance": "temporal_tracks",
                "detector_debug": self.monster_tracker.last_debug,
            },
            "a_wave": wave,
            "visual_only": visual_only,
            "registration": {
                "ok": registration.ok,
                "shift": None if not registration.ok else [round(registration.shift_x or 0.0, 3), round(registration.shift_y or 0.0, 3)],
                "confidence": round(registration.confidence, 4),
                "good_matches": registration.good_matches,
                "inliers": registration.inliers,
                "stable_fraction": round(float(np.count_nonzero(stable)) / stable.size, 4),
            },
            "quality_flags": sorted(set(flags)),
        }
        self.previous_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        self.previous_timestamp = timestamp
        if visual_only["character"]["center"] is not None:
            self.previous_character_center = list(visual_only["character"]["center"])
            self.previous_character_timestamp = timestamp
        self.frame_counter += 1
        return observation


class EventTimeline:
    def __init__(self, events: Path | str):
        self.events: list[tuple[float, str, int]] = []
        with Path(events).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("device_type") != "keyboard":
                    continue
                self.events.append((int(row["video_time_us"]) / 1_000_000.0, row["event_type"], int(row["vk_code"])))
        self.events.sort()
        self.index = 0
        self.held: set[int] = set()
        self.last_a: float | None = None
        self.last_direction: str | None = None

    def reset(self) -> None:
        self.index = 0
        self.held.clear()
        self.last_a = None
        self.last_direction = None

    def state_at(self, timestamp: float) -> dict[str, Any]:
        if self.index and timestamp < self.events[self.index - 1][0]:
            self.reset()
        while self.index < len(self.events) and self.events[self.index][0] <= timestamp:
            event_time, event_type, vk = self.events[self.index]
            if event_type == "keydown":
                self.held.add(vk)
                if vk == 65:
                    self.last_a = event_time
                if vk == 37:
                    self.last_direction = "left"
                elif vk == 39:
                    self.last_direction = "right"
            elif event_type == "keyup":
                self.held.discard(vk)
            self.index += 1
        held_direction = "left" if 37 in self.held and 39 not in self.held else "right" if 39 in self.held and 37 not in self.held else None
        return {
            "direction": held_direction or self.last_direction,
            "direction_held": held_direction is not None,
            "a_age_s": None if self.last_a is None else timestamp - self.last_a,
        }


def _copy_asset(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return destination.name


def build_session_profile(
    video_or_frames: Path | str | Sequence[np.ndarray],
    events: Path | str | None = None,
    background_assets: Path | str | None = None,
    *,
    output_dir: Path | str | None = None,
    calibration_hints: Path | str | None = None,
) -> SessionProfile:
    """Build a versioned, capture-local profile.

    For a video, optional calibration hints contain visually reviewed template
    boxes.  Without them the profile remains valid but reports that guided
    calibration is required and facing/monster recognizers abstain.
    """
    if isinstance(video_or_frames, (str, Path)):
        video = Path(video_or_frames).resolve()
        first = read_video_frame(video, frame_index=0)
        source_hash = sha256_file(video)
    else:
        video = None
        if not video_or_frames:
            raise ValueError("At least one frame is required")
        first = np.asarray(video_or_frames[0])
        source_hash = hashlib.sha256(first.tobytes()).hexdigest()
    height, width = first.shape[:2]
    if (width, height) != (1920, 1080):
        raise ValueError(f"This capture-local profile requires 1920x1080 input, got {width}x{height}")
    base = Path(output_dir).resolve() if output_dir else Path.cwd() / "session_profile"
    assets = base / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    background_dir = Path(background_assets).resolve() if background_assets else (video.parent / "background_reconstruction" if video else None)
    if background_dir is None:
        raise ValueError("background_assets is required for frame sequences")
    report = json.loads((background_dir / "report.json").read_text(encoding="utf-8"))
    for name in ("map_aligned.png", "map_aligned_stable_mask.png", "map_aligned_variability.png"):
        _copy_asset(background_dir / name, assets / name)
    reference_frame = int(report["reference"]["frame"])
    reference_time = float(report["reference"]["time_s"])
    reference_source = read_video_frame(video, frame_index=reference_frame) if video else first
    scene_top, scene_bottom, scale = 30, 945, 0.5
    reference_scene = cv2.resize(reference_source[scene_top:scene_bottom], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _write_image(assets / "reference_scene.png", reference_scene)

    character_paths: dict[str, list[str]] = {"left": [], "right": []}
    monster_paths: list[str] = []
    hint_counts = {"left": 0, "right": 0, "monster": 0}
    if calibration_hints:
        hints = json.loads(Path(calibration_hints).read_text(encoding="utf-8"))
        cache: dict[float, np.ndarray] = {}
        for item in hints.get("character", []):
            if video is None:
                continue
            timestamp = float(item["timestamp"])
            frame = cache.setdefault(timestamp, read_video_frame(video, timestamp=timestamp))
            x, y, w, h = map(int, item["bbox"])
            crop = frame[max(0, y):min(height, y + h), max(0, x):min(width, x + w)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
            edge = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 55, 140)
            facing = item["facing"]
            relative = f"assets/character_{facing}_{hint_counts[facing]:02d}.png"
            _write_image(base / relative, edge)
            character_paths[facing].append(relative)
            hint_counts[facing] += 1
        for item in hints.get("monsters", []):
            if video is None:
                continue
            timestamp = float(item["timestamp"])
            frame = cache.setdefault(timestamp, read_video_frame(video, timestamp=timestamp))
            x, y, w, h = map(int, item["bbox"])
            crop = frame[max(scene_top, y):min(scene_bottom, y + h), max(0, x):min(width, x + w)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
            edge = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 55, 140)
            relative = f"assets/monster_{hint_counts['monster']:02d}.png"
            _write_image(base / relative, edge)
            monster_paths.append(relative)
            hint_counts["monster"] += 1

    enough = hint_counts["left"] >= 30 and hint_counts["right"] >= 30
    quality = {
        "character_template_counts": {"left": hint_counts["left"], "right": hint_counts["right"]},
        "monster_template_count": hint_counts["monster"],
        "automatic_mining_status": "bootstrap_only",
        "guided_calibration_required": not enough,
        "note": "Template hints are capture-local visual anchors; keys are not stored as visual truth.",
    }
    profile = SessionProfile(
        version=PROFILE_VERSION,
        profile_id=f"pirate-ship-{source_hash[:12]}",
        source_width=width,
        source_height=height,
        scene_top=scene_top,
        scene_bottom=scene_bottom,
        scale=scale,
        reference_frame=reference_frame,
        reference_time_s=reference_time,
        reference_scene="assets/reference_scene.png",
        background_image="assets/map_aligned.png",
        stable_mask="assets/map_aligned_stable_mask.png",
        variability_image="assets/map_aligned_variability.png",
        canvas_origin_x=int(report["canvas"]["reference_origin_x"]),
        canvas_origin_y=int(report["canvas"]["reference_origin_y"]),
        character_templates=character_paths,
        monster_templates=monster_paths,
        thresholds={
            "background_residual": 34,
            "character_match": 0.29,
            "facing_margin": 0.035,
            "monster_match": 0.42,
            "monster_nms_radius": 24.0,
            "monster_min_hits": 2,
            "monster_track_gap_s": 0.55,
            "monster_supported_occlusion_s": 2.5,
            "monster_track_distance": 28.0,
            "monster_dedup_iou": 0.88,
            "monster_dedup_center_fraction": 0.22,
            "monster_dedup_size_similarity": 0.72,
            "wave_red_area": 24,
            "wave_position_margin": 18.0,
            "wave_flow_margin": 0.35,
        },
        calibration_quality=quality,
        source_hashes={
            "video_or_frames": source_hash,
            "background": sha256_file(background_dir / "map_aligned.png"),
            "events": sha256_file(Path(events)) if events else "",
        },
    )
    profile.save(base / "profile.json")
    return profile


def analyze_frame(
    image: np.ndarray | Path | str,
    profile: SessionProfile | Path | str,
    timestamp: float | None = None,
    key_state: dict[str, Any] | None = None,
    monster_detector: SpecializedMonsterDetector | None = None,
) -> dict[str, Any]:
    frame = _read_image(Path(image)) if isinstance(image, (str, Path)) else image
    return FrameAnalyzer(profile, monster_detector=monster_detector).analyze(frame, timestamp=timestamp, key_state=key_state)


def analyze_video(
    video: Path | str,
    profile: SessionProfile | Path | str,
    events: Path | str | None = None,
    monster_detector: SpecializedMonsterDetector | None = None,
) -> Iterator[dict[str, Any]]:
    video_path = Path(video)
    timestamps = video_frame_timestamps(video_path)
    timeline = EventTimeline(events) if events else None
    analyzer = FrameAnalyzer(profile, monster_detector=monster_detector)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= len(timestamps):
                # The synchronized cache ends at the intentional retained-map
                # boundary; the final minute belongs to a different map.
                break
            timestamp = timestamps[index]
            state = timeline.state_at(timestamp) if timeline else None
            observation = analyzer.analyze(frame, timestamp=timestamp, key_state=state)
            observation["frame_index"] = index
            yield observation
            index += 1
    finally:
        capture.release()
    if index != len(timestamps):
        raise RuntimeError(f"Decoded {index} frames but ffprobe reported {len(timestamps)}")
