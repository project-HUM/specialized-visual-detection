"""Command-line interface for the capture-local perception investigation."""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import json
import platform
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from perception import FrameAnalyzer, SessionProfile, build_session_profile
from perception.benchmark import build_manifest, evaluate
from perception.core import EventTimeline, sha256_file, video_frame_timestamps
from perception.render import draw_observation, write_per_frame_csv
from perception.specialized_detector import YoloMonsterDetector
from perception.monster_evaluation import evaluate_monsters
from perception.workspace import MapWorkspace
from monster_dataset.sampler import discover_recordings, sample_from_benchmark
from monster_dataset.pilot import sample_pilot
from monster_dataset.annotation_io import export_yolo, sync_splits_from_benchmark
from monster_dataset.schema import read_jsonl
from monster_dataset.validation import write_report
from monster_dataset.contact_sheet import write_contact_sheets, write_temporal_context
from monster_dataset.review_app import MonsterReviewApp
from monster_dataset.prelabel import Owlv2ProposalGenerator, TemplateProposalGenerator, prelabel_annotations


HERE = Path(__file__).resolve().parent


def _parser(workspace: MapWorkspace) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=workspace.map_id,
                        help="map workspace id under maps/ (default: rednose3)")
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("build-profile")
    profile.add_argument("--video", type=Path, default=workspace.video)
    profile.add_argument("--events", type=Path, default=workspace.events)
    profile.add_argument("--background", type=Path, default=workspace.capture_dir / "background_reconstruction")
    profile.add_argument("--hints", type=Path, default=workspace.path("calibration_hints"))
    profile.add_argument("--output", type=Path, default=workspace.path("profile_dir"))

    image = commands.add_parser("image")
    image.add_argument("input", type=Path)
    image.add_argument("--profile", type=Path, default=workspace.path("profile"))
    image.add_argument("--timestamp", type=float)
    image.add_argument("--events", type=Path)
    image.add_argument("--json", type=Path)
    image.add_argument("--annotated", type=Path)
    _add_detector_arguments(image, workspace)

    video = commands.add_parser("video")
    video.add_argument("input", type=Path, nargs="?", default=workspace.video)
    video.add_argument("--profile", type=Path, default=workspace.path("profile"))
    video.add_argument("--events", type=Path, default=workspace.events)
    video.add_argument("--start", type=float, default=0.0)
    video.add_argument("--end", type=float, default=workspace.video_end_s)
    video.add_argument("--sample-fps", type=float, default=2.0)
    video.add_argument("--jsonl", type=Path, required=True)
    video.add_argument("--csv", type=Path)
    video.add_argument("--annotated", type=Path)
    video.add_argument("--annotated-width", type=int, default=960)
    _add_detector_arguments(video, workspace)

    manifest = commands.add_parser("benchmark-manifest")
    manifest.add_argument("--events", type=Path, default=workspace.events)
    manifest.add_argument("--output", type=Path, default=workspace.path("benchmark_annotations"))
    manifest.add_argument("--count", type=int, default=360)

    extract = commands.add_parser("extract-benchmark")
    extract.add_argument("--video", type=Path, default=workspace.video)
    extract.add_argument("--annotations", type=Path, default=workspace.path("benchmark_annotations"))
    extract.add_argument("--output", type=Path, default=workspace.path("benchmark_frames"))

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--annotations", type=Path, default=workspace.path("benchmark_annotations"))
    evaluation.add_argument("--observations", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, default=workspace.path("evaluation"))
    evaluation.add_argument("--split", choices=("train", "validation", "test"), default="test")

    bench = commands.add_parser("benchmark-runtime")
    bench.add_argument("--video", type=Path, default=workspace.video)
    bench.add_argument("--profile", type=Path, default=workspace.path("profile"))
    bench.add_argument("--start", type=float, default=100.0)
    bench.add_argument("--frames", type=int, default=30)
    bench.add_argument("--output", type=Path, default=workspace.path("evaluation") / "runtime.json")
    _add_detector_arguments(bench, workspace)

    summary = commands.add_parser("summarize")
    summary.add_argument("--observations", type=Path, required=True)
    summary.add_argument("--annotations", type=Path, default=workspace.path("benchmark_annotations"))
    summary.add_argument("--output", type=Path, default=workspace.path("evaluation") / "metrics.json")

    hashes = commands.add_parser("hash-artifacts")
    hashes.add_argument("paths", nargs="+", type=Path)
    hashes.add_argument("--output", type=Path, default=workspace.map_dir / "artifact_hashes.json")

    normalize = commands.add_parser("normalize-observations")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--events", type=Path, default=workspace.events)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--csv", type=Path)

    rerender = commands.add_parser("render-observations")
    rerender.add_argument("--video", type=Path, default=workspace.video)
    rerender.add_argument("--observations", type=Path, required=True)
    rerender.add_argument("--output", type=Path, required=True)
    rerender.add_argument("--fps", type=float, default=0.5)
    rerender.add_argument("--width", type=int, default=960)

    discover = commands.add_parser("discover-video", help="list plausible capture recordings, excluding proof artifacts")
    discover.add_argument("--capture", type=Path, default=workspace.capture_dir)

    monster_sample = commands.add_parser("monster-sample", help="extract the deterministic benchmark as a canonical monster dataset")
    monster_sample.add_argument("--video", type=Path, default=workspace.video)
    monster_sample.add_argument("--benchmark", type=Path, default=workspace.path("benchmark_annotations"))
    monster_sample.add_argument("--images", type=Path, default=workspace.path("dataset_images"))
    monster_sample.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    pilot = commands.add_parser("monster-pilot-sample", help="sample an isolated pilot from direct minimap detections")
    pilot.add_argument("--video", type=Path, default=workspace.video)
    pilot.add_argument("--positions", type=Path, default=workspace.external_path("geometry", "positions")
                       if "geometry" in workspace.config else None)
    pilot.add_argument("--images", type=Path, default=workspace.path("dataset_images"))
    pilot.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    pilot.add_argument("--manifest", type=Path, default=workspace.path("pilot_manifest")
                       if "pilot_manifest" in workspace.config.get("paths", {}) else workspace.map_dir / "dataset/pilot_manifest.json")
    pilot.add_argument("--count", type=int, default=5)
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--minimum-separation", type=float, default=30.0)
    pilot.add_argument("--replace", action="store_true")
    monster_report = commands.add_parser("monster-review-report", aliases=["monster-report"], help="validate and summarize monster annotations")
    monster_report.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    monster_report.add_argument("--output", type=Path, default=workspace.path("dataset_report"))
    monster_contact = commands.add_parser("monster-contact-sheet", help="render contact sheets for annotation review")
    monster_contact.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    monster_contact.add_argument("--output", type=Path, default=workspace.path("contacts"))
    monster_contact.add_argument("--split", choices=("pilot", "train", "validation"), required=True)
    temporal = commands.add_parser("monster-temporal-context", help="render previous/current/next review strips")
    temporal.add_argument("--video", type=Path, default=workspace.video)
    temporal.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    temporal.add_argument("--split", choices=("pilot", "train", "validation"), required=True)
    temporal.add_argument("--delta", type=float, default=0.20)
    temporal.add_argument("--frame-id", action="append", help="render only selected frame IDs; repeat as needed")
    temporal.add_argument("--output", type=Path, default=workspace.path("temporal_context"))
    review = commands.add_parser("monster-review", help="interactive train/validation box review with temporal context")
    review.add_argument("--video", type=Path, default=workspace.video)
    review.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    review.add_argument("--split", choices=("all", "pilot", "train", "validation"), required=True)
    review.add_argument("--start-id")
    review.add_argument("--frame-id-prefix", action="append", default=[],
                        help="limit review to frame IDs beginning with this prefix; repeatable")
    review.add_argument("--delta", type=float, default=.20)
    review.add_argument("--queue", choices=("all", "reviewed", "pending", "needs_review", "proposal_review_required"), default="all")
    prelabel = commands.add_parser("monster-prelabel", help="generate pending visual proposals for TRAIN only")
    prelabel.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    prelabel.add_argument("--profile", type=Path, default=workspace.path("profile"))
    prelabel.add_argument("--split", choices=("train", "validation", "test"), default="train")
    prelabel.add_argument("--backend", choices=("owlv2", "template"), default="owlv2")
    prelabel.add_argument("--threshold", type=float, default=.30, help="OWLv2 text proposal threshold")
    prelabel.add_argument("--replace-pending", action="store_true",
                          help="replace existing annotations on pending TRAIN frames; never affects reviewed frames")
    prelabel.add_argument("--report", type=Path, default=workspace.path("prelabel_report"))
    monster_yolo = commands.add_parser("monster-yolo-export", help="export only reviewed canonical labels to YOLO text")
    monster_yolo.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    monster_yolo.add_argument("--output", type=Path, default=workspace.path("yolo_dataset"))
    monster_yolo.add_argument("--split", choices=("train", "validation"), required=True)
    monster_yolo.add_argument(
        "--preset-class", action="append", default=[], metavar="ID=NAME",
        help="map a box preset to a YOLO class; repeat in desired class-ID order",
    )
    sync_splits = commands.add_parser("monster-sync-splits", help="migrate split metadata without reading sealed images")
    sync_splits.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    sync_splits.add_argument("--benchmark", type=Path, default=workspace.path("benchmark_annotations"))
    monster_evaluate = commands.add_parser("monster-evaluate", help="evaluate detector and tracker separately on reviewed development labels")
    monster_evaluate.add_argument("--annotations", type=Path, default=workspace.path("annotations"))
    monster_evaluate.add_argument("--observations", type=Path, required=True)
    monster_evaluate.add_argument("--split", choices=("train", "validation"), default="validation")
    monster_evaluate.add_argument("--radius", type=float, default=64.0)
    monster_evaluate.add_argument("--output", type=Path, default=workspace.path("evaluation") / "specialized_monster")
    return parser

def _add_detector_arguments(parser: argparse.ArgumentParser, workspace: MapWorkspace) -> None:
    parser.add_argument("--monster-backend", choices=("template", "yolo"), default="template")
    parser.add_argument("--monster-weights", type=Path)
    parser.add_argument("--monster-confidence", type=float, default=.21)
    parser.add_argument(
        "--monster-nms-iou", type=float, default=.80,
        help="default YOLO NMS IoU, including Zombie (default: 0.80)",
    )
    parser.add_argument("--monster-hero-nms-iou", type=float, default=.50)
    parser.add_argument("--monster-lich-nms-iou", type=float, default=.50)
    parser.add_argument("--monster-imgsz", type=int, default=768)
    parser.add_argument("--monster-device")
    parser.add_argument(
        "--monster-box-presets", type=Path,
        default=workspace.map_dir / "dataset" / "review_settings.json",
        help="class-ordered fixed detection sizes (default: map dataset/review_settings.json)",
    )
    parser.add_argument(
        "--monster-fixed-boxes", action="store_true",
        help="replace YOLO boxes with class-ordered registered preset sizes",
    )


def _preset_classes(values: list[str]) -> dict[str, str] | None:
    if not values:
        return None
    result: dict[str, str] = {}
    for value in values:
        preset_id, separator, name = value.partition("=")
        preset_id, name = preset_id.strip(), name.strip()
        if not separator or not preset_id or not name:
            raise ValueError(f"Invalid --preset-class {value!r}; expected ID=NAME")
        if preset_id in result:
            raise ValueError(f"Duplicate --preset-class ID {preset_id!r}")
        result[preset_id] = name
    return result


def _class_box_sizes(path: Path) -> dict[int, tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    presets = payload.get("box_presets")
    if not isinstance(presets, list) or not presets:
        raise ValueError(f"{path}: box_presets must be a non-empty list")
    sizes: dict[int, tuple[float, float]] = {}
    for class_id, preset in enumerate(presets):
        try:
            width, height = float(preset["width"]), float(preset["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid box preset {class_id + 1}: {error}") from error
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}: box preset {class_id + 1} dimensions must be positive")
        sizes[class_id] = (width, height)
    return sizes

def _monster_detector(args: argparse.Namespace):
    if args.monster_backend == "template": return None
    if args.monster_weights is None: raise ValueError("--monster-weights is required for --monster-backend yolo")
    class_box_sizes = None
    if args.monster_fixed_boxes:
        if not args.monster_box_presets.is_file():
            raise ValueError(
                f"Registered YOLO box sizes not found: {args.monster_box_presets}; "
                "provide --monster-box-presets or omit --monster-fixed-boxes"
            )
        class_box_sizes = _class_box_sizes(args.monster_box_presets)
    return YoloMonsterDetector(args.monster_weights,confidence=args.monster_confidence,
                               nms_iou=args.monster_nms_iou,input_resolution=args.monster_imgsz,
                               device=args.monster_device,
                               class_nms_iou={1: args.monster_hero_nms_iou,
                                              2: args.monster_lich_nms_iou},
                               class_box_sizes=class_box_sizes,
                               box_size_reference=args.source_size if class_box_sizes else None)


def _frame_at(path: Path, timestamp: float | None) -> np.ndarray:
    if timestamp is None:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        return image
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode {path} at {timestamp}")
    return frame


def _run_video(args: argparse.Namespace) -> None:
    timestamps = video_frame_timestamps(args.input)
    start_index = bisect.bisect_left(timestamps, args.start)
    end_index = len(timestamps) if args.end is None else bisect.bisect_right(timestamps, args.end)
    minimum_step = 1.0 / args.sample_fps
    selected: list[int] = []
    next_time = args.start
    for index in range(start_index, end_index):
        if timestamps[index] + 1e-9 >= next_time:
            selected.append(index)
            next_time += minimum_step
            while next_time <= timestamps[index]:
                next_time += minimum_step
    timeline = EventTimeline(args.events) if args.events else None
    analyzer = FrameAnalyzer(args.profile, monster_detector=_monster_detector(args))
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.input}")
    writer = None
    if args.annotated:
        args.annotated.parent.mkdir(parents=True, exist_ok=True)
        out_height = round(args.source_size[1] * args.annotated_width / args.source_size[0])
        writer = cv2.VideoWriter(str(args.annotated), cv2.VideoWriter_fourcc(*"mp4v"), args.sample_fps,
                                 (args.annotated_width, out_height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open annotated output: {args.annotated}")
    observations = []
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w", encoding="utf-8") as jsonl:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        selected_set = set(selected)
        order = 0
        for frame_index in range(start_index, end_index):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {frame_index}")
            if frame_index not in selected_set:
                continue
            timestamp = timestamps[frame_index]
            state = timeline.state_at(timestamp) if timeline else None
            started = time.perf_counter()
            observation = analyzer.analyze(frame, timestamp, state)
            observation["runtime_ms"] = (time.perf_counter() - started) * 1000.0
            observation["frame_index"] = frame_index
            jsonl.write(json.dumps(observation, separators=(",", ":")) + "\n")
            observations.append(observation)
            if writer is not None:
                annotated = draw_observation(frame, observation)
                annotated = cv2.resize(annotated, (args.annotated_width, out_height), interpolation=cv2.INTER_AREA)
                writer.write(annotated)
            if order == 0 or (order + 1) % 100 == 0 or order + 1 == len(selected):
                print(f"processed {order + 1}/{len(selected)} frames", flush=True)
            order += 1
    capture.release()
    if writer is not None:
        writer.release()
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_per_frame_csv(observations, args.csv)


def _benchmark_runtime(args: argparse.Namespace) -> dict:
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    analyzer = FrameAnalyzer(args.profile, monster_detector=_monster_detector(args))
    samples = []
    for _ in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            break
        before = time.perf_counter()
        analyzer.analyze(frame)
        samples.append(time.perf_counter() - before)
    capture.release()
    if not samples:
        raise RuntimeError("No runtime benchmark frames decoded")
    try:
        import psutil
        peak_memory = psutil.Process().memory_info().rss
    except Exception:
        peak_memory = None
    result = {
        "schema_version": 1,
        "resolution": f"{args.source_size[0]}x{args.source_size[1]}",
        "frames": len(samples),
        "mean_ms": float(np.mean(samples) * 1000),
        "p50_ms": float(np.percentile(samples, 50) * 1000),
        "p95_ms": float(np.percentile(samples, 95) * 1000),
        "throughput_fps": float(1.0 / np.mean(samples)),
        "peak_process_memory_bytes": peak_memory,
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "video_sha256": sha256_file(args.video),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _extract_benchmark(args: argparse.Namespace) -> dict:
    rows = list(csv.DictReader(args.annotations.open("r", encoding="utf-8-sig", newline="")))
    timestamps = video_frame_timestamps(args.video)
    targets: dict[int, dict] = {}
    for row in rows:
        timestamp = float(row["timestamp"])
        index = bisect.bisect_left(timestamps, timestamp)
        choices = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(timestamps)]
        frame_index = min(choices, key=lambda candidate: abs(timestamps[candidate] - timestamp))
        targets[frame_index] = row
    args.output.mkdir(parents=True, exist_ok=True)
    for pattern in ("sample-*.jpg", "contact_*.jpg"):
        for old_output in args.output.glob(pattern):
            old_output.unlink()
    capture = cv2.VideoCapture(str(args.video))
    first, last = min(targets), max(targets)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    extracted: dict[str, np.ndarray] = {}
    for frame_index in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode benchmark frame {frame_index}")
        row = targets.get(frame_index)
        if row is None:
            continue
        preview = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        label = f"{row['sample_id']}  t={float(row['timestamp']):.3f}  {row['split']} / {row['category']}"
        cv2.rectangle(preview, (0, 0), (960, 34), (0, 0, 0), -1)
        cv2.putText(preview, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        filename = f"{row['sample_id']}_{float(row['timestamp']):010.3f}.jpg"
        cv2.imwrite(str(args.output / filename), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
        extracted[row["sample_id"]] = preview
        if len(extracted) == 1 or len(extracted) % 60 == 0 or len(extracted) == len(targets):
            print(f"extracted {len(extracted)}/{len(targets)} benchmark frames", flush=True)
    capture.release()
    # Twelve contact sheets (one per split/category group, with pagination for
    # the larger train groups) make visual auditing possible without seeking.
    sheets = 0
    for split in ("train", "validation", "test"):
        for category in ("clean_gameplay", "a_skill", "heavy_effects"):
            group = [row for row in rows if row["split"] == split and row["category"] == category]
            for page_start in range(0, len(group), 24):
                page = group[page_start:page_start + 24]
                cells = [cv2.resize(extracted[row["sample_id"]], (320, 180), interpolation=cv2.INTER_AREA) for row in page]
                while len(cells) < 24:
                    cells.append(np.zeros((180, 320, 3), np.uint8))
                sheet = np.vstack([np.hstack(cells[index:index + 6]) for index in range(0, 24, 6)])
                page_number = page_start // 24 + 1
                cv2.imwrite(str(args.output / f"contact_{split}_{category}_{page_number}.jpg"), sheet,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                sheets += 1
    return {"frames": len(extracted), "contact_sheets": sheets, "output": str(args.output)}


def _summarize(args: argparse.Namespace) -> dict:
    observations = [json.loads(line) for line in args.observations.read_text(encoding="utf-8").splitlines() if line.strip()]
    annotations = list(csv.DictReader(args.annotations.open("r", encoding="utf-8-sig", newline="")))
    reviewed_test = sum(row["review_status"] == "reviewed" and row["split"] == "test" for row in annotations)
    total = len(observations)
    visual_character_known = sum(row["visual_only"]["character"]["facing"] in ("left", "right") for row in observations)
    visual_wave_active = sum(row["visual_only"]["a_wave"]["state"] == "active" for row in observations)
    flag_counts: dict[str, int] = {}
    for row in observations:
        for flag in row["quality_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    monster_counts = [row["monsters"]["count"] for row in observations]
    result = {
        "schema_version": 1,
        "status": "awaiting_reviewed_ground_truth" if reviewed_test == 0 else "reviewed_ground_truth_available_run_evaluate",
        "sealed_test_reviewed_frames": reviewed_test,
        "passing_targets": {
            "monster_f1_at_64px_gte_0.85": "not_evaluated",
            "monster_count_mae_lte_0.5": "not_evaluated",
            "facing_accuracy_gte_0.95_and_coverage_gte_0.85": "not_evaluated",
            "a_wave_f1_gte_0.90_and_direction_accuracy_gte_0.95": "not_evaluated"
        },
        "diagnostic_not_ground_truth": {
            "frames": total,
            "registration_success_rate": sum(row["registration"]["ok"] for row in observations) / max(1, total),
            "visual_character_known_rate": visual_character_known / max(1, total),
            "visual_wave_active_rate": visual_wave_active / max(1, total),
            "monster_count_mean": float(np.mean(monster_counts)) if monster_counts else 0.0,
            "monster_count_p95": float(np.percentile(monster_counts, 95)) if monster_counts else 0.0,
            "quality_flag_counts": dict(sorted(flag_counts.items())),
            "warning": "These are output/abstention diagnostics, not accuracy scores. Keys and pending weak labels are not ground truth."
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _normalize_observations(args: argparse.Namespace) -> dict:
    timeline = EventTimeline(args.events)
    normalized = []
    key_flags = {
        "character_direction_key_disagreement", "wave_direction_key_disagreement", "a_key_without_visual_wave"
    }
    with args.output.open("w", encoding="utf-8") as handle:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["character"] = copy.deepcopy(row["visual_only"]["character"])
            row["a_wave"] = copy.deepcopy(row["visual_only"]["a_wave"])
            flags = [flag for flag in row["quality_flags"] if flag not in key_flags]
            FrameAnalyzer._fuse_keys(row["character"], row["a_wave"], timeline.state_at(float(row["timestamp"])), flags)
            row["quality_flags"] = sorted(set(flags))
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            normalized.append(row)
    if args.csv:
        write_per_frame_csv(normalized, args.csv)
    return {"frames": len(normalized), "output": str(args.output)}


def _render_observations(args: argparse.Namespace) -> dict:
    observations = [json.loads(line) for line in args.observations.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_frame = {int(row["frame_index"]): row for row in observations}
    first, last = min(by_frame), max(by_frame)
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    height = round(args.source_size[1] * args.width / args.source_size[0])
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {args.output}")
    written = 0
    for frame_index in range(first, last + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame_index}")
        observation = by_frame.get(frame_index)
        if observation is None:
            continue
        annotated = cv2.resize(draw_observation(frame, observation), (args.width, height), interpolation=cv2.INTER_AREA)
        writer.write(annotated)
        written += 1
        if written == 1 or written % 200 == 0 or written == len(observations):
            print(f"rendered {written}/{len(observations)} frames", flush=True)
    capture.release()
    writer.release()
    return {"frames": written, "output": str(args.output)}


def _selected_workspace(argv: list[str] | None = None) -> MapWorkspace:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--map", default=os.environ.get("SVD_MAP", "rednose3"))
    selected, _ = selector.parse_known_args(argv)
    return MapWorkspace.load(HERE, selected.map)


def main() -> int:
    workspace = _selected_workspace()
    args = _parser(workspace).parse_args()
    args.source_size = workspace.source_size
    if args.command == "build-profile":
        profile = build_session_profile(
            args.video, args.events, args.background, output_dir=args.output,
            calibration_hints=args.hints, profile_id_prefix=workspace.map_id,
            expected_source_size=workspace.source_size,
        )
        print(json.dumps({"profile": str(profile.profile_path), "quality": profile.calibration_quality}, indent=2))
    elif args.command == "image":
        frame = _frame_at(args.input, args.timestamp)
        timeline = EventTimeline(args.events) if args.events and args.timestamp is not None else None
        observation = FrameAnalyzer(args.profile, monster_detector=_monster_detector(args)).analyze(frame, args.timestamp, timeline.state_at(args.timestamp) if timeline else None)
        encoded = json.dumps(observation, indent=2)
        print(encoded)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(encoded, encoding="utf-8")
        if args.annotated:
            args.annotated.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.annotated), draw_observation(frame, observation)):
                raise RuntimeError(f"Could not write {args.annotated}")
    elif args.command == "video":
        _run_video(args)
    elif args.command == "benchmark-manifest":
        rows = build_manifest(args.events, args.output, args.count)
        print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))
    elif args.command == "extract-benchmark":
        print(json.dumps(_extract_benchmark(args), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.annotations, args.observations, args.output, split=args.split), indent=2))
    elif args.command == "benchmark-runtime":
        print(json.dumps(_benchmark_runtime(args), indent=2))
    elif args.command == "summarize":
        print(json.dumps(_summarize(args), indent=2))
    elif args.command == "hash-artifacts":
        payload = {
            "schema_version": 1,
            "algorithm": "sha256",
            "files": {
                str(path.resolve().relative_to(workspace.map_dir)).replace("\\", "/"): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in args.paths
            },
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    elif args.command == "normalize-observations":
        print(json.dumps(_normalize_observations(args), indent=2))
    elif args.command == "render-observations":
        print(json.dumps(_render_observations(args), indent=2))
    elif args.command == "discover-video":
        paths = discover_recordings(args.capture)
        print(json.dumps({"count": len(paths), "recordings": [str(path) for path in paths]}, indent=2))
    elif args.command == "monster-sample":
        items = sample_from_benchmark(args.video, args.benchmark, args.images, output_jsonl=args.annotations)
        print(json.dumps({"frames": len(items), "annotations": str(args.annotations), "images": str(args.images)}, indent=2))
    elif args.command == "monster-pilot-sample":
        if args.positions is None:
            raise ValueError(f"Map {workspace.map_id!r} has no direct geometry positions configured")
        print(json.dumps(sample_pilot(
            args.video, args.positions, args.images, args.annotations, args.manifest,
            repo_root=workspace.repo_root, map_id=workspace.map_id,
            minimap_crop=workspace.minimap_crop, source_size=workspace.source_size,
            count=args.count, seed=args.seed,
            minimum_separation_s=args.minimum_separation, replace=args.replace,
        ), indent=2))
    elif args.command in {"monster-review-report", "monster-report"}:
        print(json.dumps(write_report(read_jsonl(args.annotations), args.output,
                                      source_size=workspace.source_size), indent=2))
    elif args.command == "monster-contact-sheet":
        outputs = write_contact_sheets(read_jsonl(args.annotations), args.output, split=args.split,
                                       source_size=workspace.source_size,
                                       annotations_path=args.annotations)
        print(json.dumps({"sheets": len(outputs), "output": str(args.output)}, indent=2))
    elif args.command == "monster-temporal-context":
        outputs = write_temporal_context(read_jsonl(args.annotations), args.video, args.output,
                                         split=args.split, delta_s=args.delta,
                                         frame_ids=set(args.frame_id) if args.frame_id else None)
        print(json.dumps({"frames": len(outputs), "split": args.split, "output": str(args.output)}, indent=2))
    elif args.command == "monster-review":
        MonsterReviewApp(args.annotations,args.video,split=args.split,start_id=args.start_id,
                         delta_s=args.delta,queue=args.queue,
                         frame_id_prefixes=tuple(args.frame_id_prefix),
                         source_size=workspace.source_size).run()
    elif args.command == "monster-prelabel":
        if args.split != "train":
            raise ValueError("Automatic pre-labeling is restricted to TRAIN; validation is manual and test is sealed")
        if args.backend == "owlv2" and not workspace.prompts:
            raise ValueError(f"Map {workspace.map_id!r} has no reviewed OWLv2 prompts configured")
        generator = (Owlv2ProposalGenerator(
            threshold=args.threshold, prompts=workspace.prompts,
            source_size=workspace.source_size, fixed_ui_rects=workspace.fixed_ui_rects)
            if args.backend == "owlv2" else TemplateProposalGenerator(
                args.profile, fixed_ui_rects=workspace.fixed_ui_rects))
        print(json.dumps(prelabel_annotations(read_jsonl(args.annotations), args.annotations,
                                              split=args.split, generator=generator,
                                              replace_pending=args.replace_pending,
                                              report_path=args.report,
                                              progress=lambda done, total: print(
                                                  f"pre-labeled {done}/{total} TRAIN frames", file=sys.stderr, flush=True
                                              )), indent=2))
    elif args.command == "monster-yolo-export":
        print(json.dumps(export_yolo(read_jsonl(args.annotations), args.output, split=args.split,
                                     source_size=workspace.source_size,
                                     image_root=workspace.repo_root,
                                     preset_classes=_preset_classes(args.preset_class)), indent=2))
    elif args.command == "monster-sync-splits":
        print(json.dumps(sync_splits_from_benchmark(read_jsonl(args.annotations),args.benchmark,args.annotations),indent=2))
    elif args.command == "monster-evaluate":
        print(json.dumps(evaluate_monsters(args.annotations,args.observations,args.output,split=args.split,radius=args.radius),indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
