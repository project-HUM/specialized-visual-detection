"""Observation drawing helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def draw_observation(frame: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
    canvas = frame.copy()
    character = observation["character"]
    if character["center"] is not None:
        point = tuple(int(round(value)) for value in character["center"])
        color = (80, 220, 80) if character["facing"] != "unknown" else (0, 190, 255)
        cv2.circle(canvas, point, 34, color, 3)
        cv2.putText(canvas, f"character {character['facing']} {character['confidence']:.2f}",
                    (point[0] - 100, point[1] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    tracks = observation["monsters"].get("tracks", observation["monsters"].get("centers", []))
    for index, monster in enumerate(tracks):
        point = tuple(int(round(value)) for value in monster["center"])
        state = monster.get("state", "visible")
        color = (255, 170, 30) if state == "visible" else (0, 190, 255) if state == "occluded" else (150, 150, 150)
        cv2.circle(canvas, point, 28, color, 2)
        cv2.putText(canvas, f"M{monster.get('track_id', index + 1)} {state}", (point[0] - 38, point[1] - 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    wave = observation["a_wave"]
    wave_color = (30, 30, 240) if wave["state"] == "active" else (170, 170, 170)
    lines = [
        f"t={observation['timestamp'] if observation['timestamp'] is not None else '-'}",
        f"monsters visual={observation['monsters'].get('visual_detection_count', observation['monsters']['count'])} estimated={observation['monsters'].get('estimated_count', observation['monsters']['count'])}",
        f"A-wave={wave['state']}/{wave['direction']} {wave['confidence']:.2f} ({wave['provenance']})",
        f"registration={'ok' if observation['registration']['ok'] else 'FAILED'} {observation['registration']['confidence']:.2f}",
    ]
    y = 380
    for line in lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, wave_color, 2, cv2.LINE_AA)
        y += 28
    if observation["quality_flags"]:
        flag_text = ", ".join(observation["quality_flags"][:4])
        cv2.putText(canvas, flag_text, (20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 190, 255), 2, cv2.LINE_AA)
    return canvas


def write_per_frame_csv(observations: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "frame_index", "timestamp", "character_x", "character_y", "character_facing", "character_confidence",
        "monster_count", "visual_detection_count", "estimated_monster_count", "monster_centers", "a_wave_state", "a_wave_direction", "a_wave_confidence",
        "a_wave_provenance", "registration_ok", "registration_confidence", "quality_flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in observations:
            center = item["character"]["center"]
            writer.writerow({
                "frame_index": item["frame_index"],
                "timestamp": item["timestamp"],
                "character_x": "" if center is None else center[0],
                "character_y": "" if center is None else center[1],
                "character_facing": item["character"]["facing"],
                "character_confidence": item["character"]["confidence"],
                "monster_count": item["monsters"]["count"],
                "visual_detection_count": item["monsters"].get("visual_detection_count", item["monsters"]["count"]),
                "estimated_monster_count": item["monsters"].get("estimated_count", item["monsters"]["count"]),
                "monster_centers": json.dumps([m["center"] for m in item["monsters"]["centers"]]),
                "a_wave_state": item["a_wave"]["state"],
                "a_wave_direction": item["a_wave"]["direction"],
                "a_wave_confidence": item["a_wave"]["confidence"],
                "a_wave_provenance": item["a_wave"]["provenance"],
                "registration_ok": item["registration"]["ok"],
                "registration_confidence": item["registration"]["confidence"],
                "quality_flags": ";".join(item["quality_flags"]),
            })
