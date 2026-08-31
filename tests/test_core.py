from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from perception.core import (
    FrameAnalyzer,
    MapRegistrar,
    SessionProfile,
    _scene,
    probe_frame_timestamps,
)
from render_background_subtraction import remove_fixed_ui


def _profile(tmp_path: Path) -> tuple[SessionProfile, np.ndarray]:
    rng = np.random.default_rng(17)
    source = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    scene = cv2.resize(source[30:945], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    assets = tmp_path / "assets"
    assets.mkdir()
    canvas = np.zeros((685, 1268, 3), np.uint8)
    canvas[220:220 + scene.shape[0], 1:1 + scene.shape[1]] = scene
    stable = np.zeros(canvas.shape[:2], np.uint8)
    stable[220:220 + scene.shape[0], 1:1 + scene.shape[1]] = 255
    variability = np.zeros_like(stable)
    cv2.imwrite(str(assets / "reference.png"), scene)
    cv2.imwrite(str(assets / "background.png"), canvas)
    cv2.imwrite(str(assets / "stable.png"), stable)
    cv2.imwrite(str(assets / "variability.png"), variability)
    profile = SessionProfile(
        version=1,
        profile_id="test",
        source_width=1920,
        source_height=1080,
        scene_top=30,
        scene_bottom=945,
        scale=0.5,
        reference_frame=0,
        reference_time_s=0.0,
        reference_scene="assets/reference.png",
        background_image="assets/background.png",
        stable_mask="assets/stable.png",
        variability_image="assets/variability.png",
        canvas_origin_x=1,
        canvas_origin_y=220,
        thresholds={"background_residual": 34},
    )
    profile.save(tmp_path / "profile.json")
    return profile, source


def test_profile_round_trip(tmp_path: Path):
    profile, _ = _profile(tmp_path)
    loaded = SessionProfile.load(tmp_path / "profile.json")
    assert loaded.profile_id == profile.profile_id
    assert loaded.asset(loaded.background_image).is_file()
    payload = json.loads((tmp_path / "profile.json").read_text())
    assert "profile_path" not in payload


def test_resolution_rejection(tmp_path: Path):
    profile, _ = _profile(tmp_path)
    with pytest.raises(ValueError, match="Resolution mismatch"):
        _scene(np.zeros((720, 1280, 3), np.uint8), profile)


def test_vfr_timestamps_are_read_verbatim(monkeypatch, tmp_path: Path):
    output = "0.000000\n0.033000\n0.071500\n0.102000\n"
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout=output))
    assert probe_frame_timestamps(tmp_path / "vfr.mp4") == [0.0, 0.033, 0.0715, 0.102]


def test_ui_mask_uses_source_coordinate_rectangles(tmp_path: Path):
    profile, _ = _profile(tmp_path)
    registrar = MapRegistrar(profile)
    assert registrar.valid_mask[20, 20] == 0  # minimap
    assert registrar.valid_mask[200, 300] == 255
    assert registrar.valid_mask[-10, 100] == 0  # chat crop overlap


def test_residual_cutout_removes_fixed_ui(tmp_path: Path):
    profile, _ = _profile(tmp_path)
    registrar = MapRegistrar(profile)
    residual = np.ones(registrar.valid_mask.shape, dtype=bool)
    cleaned = remove_fixed_ui(residual, registrar.valid_mask)
    assert not cleaned[20, 20]  # minimap
    assert cleaned[200, 300]  # gameplay
    assert not cleaned[-10, 100]  # bottom HUD


def test_residual_cutout_rejects_mismatched_ui_mask():
    with pytest.raises(ValueError, match="shape mismatch"):
        remove_fixed_ui(np.ones((4, 4), dtype=bool), np.ones((3, 4), np.uint8))


def test_registration_and_background_transform(tmp_path: Path):
    profile, source = _profile(tmp_path)
    analyzer = FrameAnalyzer(profile)
    scene = _scene(source, profile)
    registration = analyzer.registrar.register(scene)
    assert registration.ok
    assert abs(registration.shift_x or 0) < 0.5
    assert abs(registration.shift_y or 0) < 0.5
    residual, stable = analyzer._registered_residual(scene, registration)
    assert np.count_nonzero(residual) == 0
    assert np.count_nonzero(stable) > 0


def test_failed_registration_returns_unknown_wave(tmp_path: Path):
    profile, _ = _profile(tmp_path)
    observation = FrameAnalyzer(profile).analyze(np.zeros((1080, 1920, 3), np.uint8))
    assert observation["registration"]["ok"] is False
    assert observation["a_wave"]["state"] == "unknown"
    assert "registration_failed" in observation["quality_flags"]


def test_single_image_abstains_without_character_templates(tmp_path: Path):
    profile, source = _profile(tmp_path)
    observation = FrameAnalyzer(profile).analyze(source)
    assert observation["character"]["facing"] == "unknown"
    assert observation["character"]["center"] is None
    assert observation["a_wave"]["state"] == "unknown"
    assert "wave_character_missing" in observation["quality_flags"]


def test_direction_disagreement_is_not_silently_overwritten():
    character = {"facing": "left", "confidence": 0.9, "provenance": "visual"}
    wave = {"state": "active", "direction": "left", "confidence": 0.9, "provenance": "visual"}
    flags: list[str] = []
    FrameAnalyzer._fuse_keys(character, wave, {"direction": "right", "direction_held": True, "a_age_s": 0.2}, flags)
    assert character["facing"] == "left"
    assert wave["direction"] == "left"
    assert "character_direction_key_disagreement" in flags
    assert "wave_direction_key_disagreement" in flags


def test_key_only_wave_evidence_remains_unknown():
    character = {"facing": "unknown", "confidence": 0.0, "provenance": "visual"}
    wave = {"state": "inactive", "direction": "inactive", "confidence": 0.8, "provenance": "visual"}
    flags: list[str] = []
    FrameAnalyzer._fuse_keys(character, wave, {"direction": "right", "direction_held": True, "a_age_s": 0.1}, flags)
    assert wave["state"] == "unknown"
    assert wave["provenance"] == "keys"
    assert "a_key_without_visual_wave" in flags


def test_stale_last_direction_is_not_fused_as_visible_facing():
    character = {"facing": "unknown", "confidence": 0.0, "provenance": "visual"}
    wave = {"state": "inactive", "direction": "inactive", "confidence": 0.8, "provenance": "visual"}
    flags: list[str] = []
    FrameAnalyzer._fuse_keys(character, wave, {"direction": "right", "direction_held": False, "a_age_s": None}, flags)
    assert character["facing"] == "unknown"
    assert character["provenance"] == "visual"
