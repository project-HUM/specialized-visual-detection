"""Render high-recall YOLO proposals for the unreviewed portion of Forest batch 4."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monster_dataset.schema import read_jsonl
from tools.render_tiny_yolo_results import render


MAP_DIR = ROOT / "maps/forest-of-dead-trees-2"
ANNOTATIONS = MAP_DIR / "dataset/annotations.jsonl"
RUN_DIR = MAP_DIR / "dataset/runs/codex-manual-batch-4"
PREVIEW_MANIFEST = RUN_DIR / "proposal-preview-manifest.json"
OUTPUT = RUN_DIR / "proposal-preview-conf005"
WEIGHTS = MAP_DIR / "dataset/runs/combined-v2/training-640/weights/best.pt"
PRESETS = MAP_DIR / "dataset/review_settings.json"
PREFIX = "codex-sample4-hardcase-"


def main() -> int:
    frames = sorted(
        (item for item in read_jsonl(ANNOTATIONS)
         if item.frame_id.startswith(PREFIX) and int(item.frame_id.rsplit("-", 1)[1]) >= 2),
        key=lambda item: item.frame_id,
    )
    if len(frames) != 13:
        raise ValueError(f"Expected 13 batch-4 targets, found {len(frames)}")
    manifest = {
        "schema": "specialized-visual-detection.proposal-preview.v1",
        "test_frames": [
            {
                "id": item.frame_id,
                "time_s": item.timestamp,
                "kind": "batch4_manual_prelabelling",
                "image": item.image_path,
            }
            for item in frames
        ],
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = render(
        WEIGHTS, PREVIEW_MANIFEST, OUTPUT,
        confidence=.05, imgsz=640, device="cpu", box_presets=PRESETS,
    )
    print(json.dumps({
        "frames": len(result["frames"]),
        "counts_by_kind": result["counts_by_kind"],
        "contact_sheet": result["contact_sheet"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
