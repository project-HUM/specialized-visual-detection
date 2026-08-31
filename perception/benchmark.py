"""Deterministic benchmark manifests and metrics for reviewed annotations."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .core import EventTimeline, sha256_file


SPLITS = (
    ("train", 0.0, 1298.4),
    ("validation", 1298.4, 1731.2),
    ("test", 1731.2, 2163.8),
)


def _split_at(timestamp: float) -> str | None:
    for name, start, end in SPLITS:
        if start <= timestamp < end:
            return name
    return None


def _a_events(events_path: Path) -> list[tuple[float, str]]:
    timeline = EventTimeline(events_path)
    output: list[tuple[float, str]] = []
    held: set[int] = set()
    last_direction = "unknown"
    for timestamp, event_type, vk in timeline.events:
        if event_type == "keydown" and vk == 65:
            direction = "left" if 37 in held and 39 not in held else "right" if 39 in held and 37 not in held else last_direction
            output.append((timestamp, direction))
        if vk in (37, 39):
            if event_type == "keydown":
                held.add(vk)
                last_direction = "left" if vk == 37 else "right"
            elif event_type == "keyup":
                held.discard(vk)
    return output


def _non_movement_keydowns(events_path: Path) -> np.ndarray:
    times: list[float] = []
    with events_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("device_type") != "keyboard" or row.get("event_type") != "keydown":
                continue
            if int(row["vk_code"]) not in (37, 38, 39, 40):
                times.append(int(row["video_time_us"]) / 1_000_000.0)
    return np.asarray(times)


def _spread(items: list[tuple[float, str]], count: int) -> list[tuple[float, str]]:
    if count <= 0:
        return []
    if len(items) < count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[int(index)] for index in indices]


def build_manifest(events_path: Path, output_path: Path, count: int = 360) -> list[dict[str, Any]]:
    """Create 360 non-adjacent candidates without crossing split time ranges.

    Candidate labels are explicitly weak until a human sets review_status to
    ``reviewed``.  This prevents synchronized keys from becoming visual truth.
    """
    if count % 3:
        raise ValueError("count must be divisible by three")
    events = _a_events(events_path)
    action_times = _non_movement_keydowns(events_path)
    chosen: list[dict[str, Any]] = []

    def add(timestamp: float, category: str, weak_state: str, weak_direction: str) -> bool:
        split = _split_at(timestamp)
        if split is None or any(abs(timestamp - row["timestamp"]) < 0.55 for row in chosen):
            return False
        chosen.append({
            "sample_id": f"sample-{len(chosen):04d}",
            "timestamp": round(timestamp, 6),
            "split": split,
            "category": category,
            "review_status": "pending",
            "character_x": "",
            "character_y": "",
            "character_visible": "",
            "character_facing": "",
            "monster_centers": "",
            "a_wave_state": "",
            "a_wave_direction": "",
            "weak_a_wave_state": weak_state,
            "weak_a_wave_direction": weak_direction,
            "notes": "",
        })
        return True

    split_totals = {"train": count * 3 // 5, "validation": count // 5, "test": count // 5}
    cast_times = np.asarray([item[0] for item in events])
    for split_name, split_start, split_end in SPLITS:
        category_target = split_totals[split_name] // 3
        split_events = [(t, d) for t, d in events if split_start <= t + 0.32 < split_end]
        # Direction-stratified and time-spread cast frames.  Preserve every
        # feasible scarce left cast, then fill with evenly spaced right casts.
        left_casts = [item for item in split_events if item[1] == "left"]
        right_casts = [item for item in split_events if item[1] != "left"]
        left_target = min(len(left_casts), max(1, category_target // 2))
        preferred_casts = sorted(_spread(left_casts, left_target) + _spread(right_casts, category_target - left_target))
        preferred_set = set(preferred_casts)
        ordered_casts = preferred_casts + [item for item in split_events if item not in preferred_set]
        for timestamp, direction in ordered_casts:
            if sum(row["split"] == split_name and row["category"] == "a_skill" for row in chosen) >= category_target:
                break
            add(timestamp + 0.32, "a_skill", "active", direction)
        if sum(row["split"] == split_name and row["category"] == "a_skill" for row in chosen) < category_target:
            raise RuntimeError(f"Not enough A events in {split_name} for balanced sampling")

        # Later effect frames are candidate-mining only, not ground truth.
        # Reverse ordering avoids selecting exactly the same subset first.
        heavy_preferred = list(reversed(_spread(split_events, category_target)))
        heavy_set = set(heavy_preferred)
        heavy_candidates = heavy_preferred + [item for item in reversed(split_events) if item not in heavy_set]
        for timestamp, direction in heavy_candidates:
            if sum(row["split"] == split_name and row["category"] == "heavy_effects" for row in chosen) >= category_target:
                break
            add(timestamp + 0.90, "heavy_effects", "active", direction)
        if sum(row["split"] == split_name and row["category"] == "heavy_effects" for row in chosen) < category_target:
            raise RuntimeError(f"Not enough heavy-effect candidates in {split_name}")

        # Uniform clean candidates within this complete time range, excluding
        # all cast vicinities and already chosen frames.
        grid = np.linspace(split_start + 2.0, split_end - 2.0, category_target * 40)
        rng = np.random.default_rng(20260831 + list(split_totals).index(split_name))
        for timestamp in grid[rng.permutation(len(grid))]:
            if sum(row["split"] == split_name and row["category"] == "clean_gameplay" for row in chosen) >= category_target:
                break
            if np.min(np.abs(cast_times - timestamp)) < 1.5:
                continue
            if len(action_times) and np.min(np.abs(action_times - timestamp)) < 0.85:
                continue
            add(float(timestamp), "clean_gameplay", "inactive", "inactive")
        if sum(row["split"] == split_name and row["category"] == "clean_gameplay" for row in chosen) < category_target:
            raise RuntimeError(f"Not enough clean candidates in {split_name}")
    if len(chosen) != count:
        raise RuntimeError(f"Could only create {len(chosen)} of {count} benchmark candidates")
    chosen.sort(key=lambda row: row["timestamp"])
    for index, row in enumerate(chosen):
        row["sample_id"] = f"sample-{index:04d}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chosen[0]))
        writer.writeheader()
        writer.writerows(chosen)
    manifest = {
        "schema_version": 1,
        "count": len(chosen),
        "minimum_separation_s": 0.55,
        "split_ranges": [{"name": n, "start": s, "end": e} for n, s, e in SPLITS],
        "categories": dict(Counter(row["category"] for row in chosen)),
        "splits": dict(Counter(row["split"] for row in chosen)),
        "annotations_sha256": sha256_file(output_path),
        "warning": "weak_* columns are candidate-mining evidence only; evaluation uses review_status=reviewed.",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return chosen


def _parse_centers(value: str) -> list[tuple[float, float]]:
    if not value.strip():
        return []
    parsed = json.loads(value)
    return [(float(point[0]), float(point[1])) for point in parsed]


def _greedy_matches(truth: list[tuple[float, float]], predicted: list[tuple[float, float]], radius: float = 64.0) -> int:
    candidates = sorted(
        (math.dist(a, b), i, j) for i, a in enumerate(truth) for j, b in enumerate(predicted) if math.dist(a, b) <= radius
    )
    used_truth: set[int] = set()
    used_predicted: set[int] = set()
    matches = 0
    for _, i, j in candidates:
        if i not in used_truth and j not in used_predicted:
            used_truth.add(i)
            used_predicted.add(j)
            matches += 1
    return matches


def evaluate(annotations_path: Path, observations_path: Path, output_dir: Path, *, split: str = "test") -> dict[str, Any]:
    annotations = [row for row in csv.DictReader(annotations_path.open(encoding="utf-8-sig"))
                   if row["review_status"] == "reviewed" and row["split"] == split]
    observations = [json.loads(line) for line in observations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not annotations:
        raise RuntimeError(f"No reviewed {split} annotations; weak labels are intentionally rejected")
    by_time = sorted(observations, key=lambda row: row["timestamp"])
    pairs = []
    for truth in annotations:
        timestamp = float(truth["timestamp"])
        observation = min(by_time, key=lambda row: abs(float(row["timestamp"]) - timestamp))
        if abs(float(observation["timestamp"]) - timestamp) > 0.08:
            continue
        pairs.append((truth, observation))
    if not pairs:
        raise RuntimeError("No observations aligned within 80 ms of reviewed annotations")

    def score(source: str) -> dict[str, Any]:
        char_total = char_known = char_correct = 0
        wave_tp = wave_fp = wave_fn = direction_total = direction_correct = 0
        direction_by_class = {"left": [0, 0], "right": [0, 0]}
        monster_tp = monster_fp = monster_fn = 0
        count_errors: list[int] = []
        confusion = Counter()
        for truth, observation in pairs:
            result = observation if source == "fused" else {**observation, **observation["visual_only"]}
            truth_facing = truth["character_facing"]
            predicted_facing = result["character"]["facing"]
            if truth.get("character_visible", "").lower() == "true" and truth_facing in ("left", "right"):
                char_total += 1
                if predicted_facing in ("left", "right"):
                    char_known += 1
                    char_correct += int(predicted_facing == truth_facing)
                confusion[(truth_facing, predicted_facing)] += 1
            truth_active = truth["a_wave_state"] == "active"
            predicted_active = result["a_wave"]["state"] == "active"
            wave_tp += int(truth_active and predicted_active)
            wave_fp += int(not truth_active and predicted_active)
            wave_fn += int(truth_active and not predicted_active)
            truth_direction = truth["a_wave_direction"]
            if truth_active and predicted_active and truth_direction in ("left", "right"):
                direction_total += 1
                ok = result["a_wave"]["direction"] == truth_direction
                direction_correct += int(ok)
                direction_by_class[truth_direction][1] += 1
                direction_by_class[truth_direction][0] += int(ok)
            truth_centers = _parse_centers(truth["monster_centers"])
            predicted_centers = [tuple(item["center"]) for item in result["monsters"]["centers"]]
            matched = _greedy_matches(truth_centers, predicted_centers)
            monster_tp += matched
            monster_fn += len(truth_centers) - matched
            monster_fp += len(predicted_centers) - matched
            count_errors.append(abs(len(truth_centers) - len(predicted_centers)))
        precision = monster_tp / max(1, monster_tp + monster_fp)
        recall = monster_tp / max(1, monster_tp + monster_fn)
        wave_precision = wave_tp / max(1, wave_tp + wave_fp)
        wave_recall = wave_tp / max(1, wave_tp + wave_fn)
        return {
            "frames": len(pairs),
            "character": {
                "accuracy": char_correct / max(1, char_known),
                "coverage": char_known / max(1, char_total),
                "evaluated_visible": char_total,
                "confusion": {f"{a}->{b}": n for (a, b), n in sorted(confusion.items())},
            },
            "monsters": {
                "f1_at_64px": 2 * precision * recall / max(1e-12, precision + recall),
                "precision": precision,
                "recall": recall,
                "count_mae": float(np.mean(count_errors)),
                "count_errors": count_errors,
            },
            "a_wave": {
                "event_f1": 2 * wave_precision * wave_recall / max(1e-12, wave_precision + wave_recall),
                "event_precision": wave_precision,
                "event_recall": wave_recall,
                "direction_accuracy": direction_correct / max(1, direction_total),
                "direction_by_class": {name: {"correct": values[0], "total": values[1], "accuracy": values[0] / max(1, values[1])}
                                       for name, values in direction_by_class.items()},
            },
        }

    metrics = {"schema_version": 1, "split": split, "reviewed_pairs": len(pairs), "visual_only": score("visual"), "key_assisted": score("fused")}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    errors = metrics["visual_only"]["monsters"]["count_errors"]
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=np.arange(-0.5, max(errors + [1]) + 1.5), rwidth=0.85)
    plt.xlabel("absolute count error")
    plt.ylabel("frames")
    plt.tight_layout()
    plt.savefig(output_dir / "count_error.png", dpi=150)
    plt.close()
    return metrics
