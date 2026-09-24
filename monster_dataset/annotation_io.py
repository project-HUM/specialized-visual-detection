"""Canonical annotation import and strict split-aware detector export."""
from __future__ import annotations
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from .schema import FrameAnnotation
from .schema import write_jsonl


def write_jsonl_with_backup(items: list[FrameAnnotation], path: Path) -> None:
    """Atomically replace annotations while retaining one recoverable backup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    backup = path.with_suffix(path.suffix + ".bak")
    write_jsonl(items, temporary)
    if path.is_file():
        shutil.copy2(path, backup)
    os.replace(temporary, path)

def from_benchmark_csv(path: Path, image_root: Path) -> list[FrameAnnotation]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    return [FrameAnnotation(frame_id=row["sample_id"], timestamp=float(row["timestamp"]),
                            image_path=str(image_root / f"{row['sample_id']}_{float(row['timestamp']):010.3f}.jpg"),
                            split=row["split"], category=row.get("category", "unspecified"),
                            review_status=row.get("review_status", "pending"), notes=row.get("notes", "")) for row in rows]

def _image_path(image_path: str, image_root: Path) -> Path:
    candidate = Path(image_path)
    return candidate if candidate.is_absolute() else image_root / candidate


def validate_for_export(items: list[FrameAnnotation], split: str, *,
                        source_size: tuple[int, int] = (1920, 1080),
                        image_root: Path = Path.cwd()) -> list[FrameAnnotation]:
    if split not in {"train", "validation"}:
        raise ValueError("Detector development export permits only train or validation; sealed test is protected")
    selected = [item for item in items if item.split == split]
    if not selected:
        raise ValueError(f"No annotations found for split {split!r}")
    failures: list[str] = []
    for item in selected:
        if item.review_status != "reviewed":
            failures.append(f"{item.frame_id}: review_status={item.review_status}")
        if any(monster.review_required for monster in item.monsters):
            failures.append(f"{item.frame_id}: contains review_required monster")
        failures.extend(f"{item.frame_id}: {error}" for error in item.validate(*source_size))
        if not _image_path(item.image_path, image_root).is_file():
            failures.append(f"{item.frame_id}: image does not exist: {item.image_path}")
    if failures:
        preview = "\n".join(failures[:20])
        suffix = f"\n... and {len(failures)-20} more" if len(failures) > 20 else ""
        raise ValueError(f"Refusing {split} export; annotations are not training-ready:\n{preview}{suffix}")
    return selected

def export_yolo(items: list[FrameAnnotation], output_dir: Path, *, split: str,
                source_size: tuple[int, int] = (1920, 1080),
                image_root: Path = Path.cwd(),
                preset_classes: dict[str, str] | None = None) -> dict[str, object]:
    selected = validate_for_export(items, split, source_size=source_size, image_root=image_root)
    if preset_classes:
        if any(not preset_id or not name for preset_id, name in preset_classes.items()):
            raise ValueError("Preset class IDs and names must be non-empty")
        if len(set(preset_classes.values())) != len(preset_classes):
            raise ValueError("Preset class names must be unique")
        class_ids = {preset_id: index for index, preset_id in enumerate(preset_classes)}
        missing = [
            f"{item.frame_id}: monster[{index}] box_preset={monster.box_preset!r}"
            for item in selected
            for index, monster in enumerate(item.monsters)
            if monster.box_preset not in class_ids
        ]
        if missing:
            raise ValueError(
                "Refusing multi-class export; every monster must use a mapped box preset:\n" +
                "\n".join(missing[:20])
            )
        names = list(preset_classes.values())
    else:
        class_ids = {}
        names = ["monster"]
    labels_dir, images_dir = output_dir / "labels" / split, output_dir / "images" / split
    labels_dir.mkdir(parents=True, exist_ok=True); images_dir.mkdir(parents=True, exist_ok=True)
    instances = 0
    class_counts: Counter[str] = Counter()
    for item in selected:
        lines: list[str] = []
        for monster in item.monsters:
            x1, y1, x2, y2 = monster.bbox_xyxy
            width, height = source_size
            x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
            y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
            class_id = class_ids[monster.box_preset] if preset_classes else 0
            class_name = names[class_id]
            lines.append(f"{class_id} {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
            instances += 1
            class_counts[class_name] += 1
        (labels_dir / f"{item.frame_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        source = _image_path(item.image_path, image_root)
        shutil.copy2(source, images_dir / f"{item.frame_id}{source.suffix.lower()}")
    yaml_names = "".join(f"  {index}: {json.dumps(name)}\n" for index, name in enumerate(names))
    (output_dir / "dataset.yaml").write_text(
        f"path: {output_dir.resolve().as_posix()}\ntrain: images/train\nval: images/validation\nnames:\n{yaml_names}",
        encoding="utf-8",
    )
    return {"split": split, "frames": len(selected), "instances": instances,
            "classes": dict(sorted(class_counts.items())), "output": str(output_dir)}

def sync_splits_from_benchmark(items: list[FrameAnnotation], benchmark_csv: Path,
                               output_path: Path) -> dict[str, int]:
    """Migrate split metadata without decoding or revealing any test pixels."""
    split_by_id = {row["sample_id"]: row["split"] for row in csv.DictReader(
        benchmark_csv.open("r", encoding="utf-8-sig", newline=""))}
    missing = [item.frame_id for item in items if item.frame_id not in split_by_id]
    if missing:
        raise ValueError(f"Benchmark is missing {len(missing)} annotation frame IDs")
    for item in items:
        item.split = split_by_id[item.frame_id]
        local_image = output_path.parent / "images" / Path(item.image_path).name
        if local_image.is_file():
            try:
                item.image_path = local_image.relative_to(output_path.parent.parent).as_posix()
            except ValueError:
                item.image_path = str(local_image)
    write_jsonl(items, output_path)
    return {split: sum(item.split == split for item in items) for split in ("train", "validation", "test")}
