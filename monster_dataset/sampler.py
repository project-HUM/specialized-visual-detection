from __future__ import annotations
import bisect, csv, json
from pathlib import Path
from typing import Any
from perception.core import video_frame_timestamps, read_video_frame
from .annotation_io import from_benchmark_csv
from .schema import FrameAnnotation, write_jsonl

def discover_recordings(capture_dir: Path) -> list[Path]:
    candidates=[]
    for path in capture_dir.rglob("*"):
        if path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov"} and "proof" not in path.parts and path.stat().st_size > 1_000_000:
            candidates.append(path)
    return sorted(candidates)

def sample_from_benchmark(video: Path, benchmark_csv: Path, output_dir: Path, *, output_jsonl: Path | None = None) -> list[FrameAnnotation]:
    output_dir.mkdir(parents=True,exist_ok=True)
    items=from_benchmark_csv(benchmark_csv, output_dir)
    timestamps=video_frame_timestamps(video)
    # Decode by nearest presentation timestamp, not nominal FPS.
    import cv2
    capture=cv2.VideoCapture(str(video))
    for item in items:
        target=bisect.bisect_left(timestamps,item.timestamp); choices=[i for i in (target-1,target) if 0<=i<len(timestamps)]
        index=min(choices,key=lambda i:abs(timestamps[i]-item.timestamp)); capture.set(cv2.CAP_PROP_POS_FRAMES,index); ok,frame=capture.read()
        if not ok: raise RuntimeError(f"Could not decode benchmark frame {index}")
        path=output_dir/f"{item.frame_id}_{item.timestamp:010.3f}.jpg"; cv2.imwrite(str(path),cv2.resize(frame,(960,540)),[cv2.IMWRITE_JPEG_QUALITY,92])
        try: item.image_path=path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError: item.image_path=str(path.resolve())
    capture.release()
    if output_jsonl: write_jsonl(items,output_jsonl)
    return items
