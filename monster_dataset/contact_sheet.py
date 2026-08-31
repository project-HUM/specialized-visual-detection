from __future__ import annotations
from pathlib import Path
import bisect
import cv2, numpy as np
from .schema import FrameAnnotation
from perception.core import read_video_frame, video_frame_timestamps

def write_contact_sheets(items: list[FrameAnnotation], output_dir: Path, *, split: str,
                         columns: int=5, cell_size: tuple[int,int]=(320,180)) -> list[Path]:
    if split not in {"train","validation"}: raise ValueError("Review sheets are limited to train and validation")
    items=[item for item in items if item.split==split]
    output_dir.mkdir(parents=True,exist_ok=True); outputs=[]
    for start in range(0,len(items),columns*4):
        cells=[]
        for item in items[start:start+columns*4]:
            image=cv2.imread(item.image_path); image=cv2.resize(image,cell_size) if image is not None else np.zeros((cell_size[1],cell_size[0],3),np.uint8)
            cv2.rectangle(image,(0,0),(cell_size[0],22),(0,0,0),-1); cv2.putText(image,f"{item.frame_id} {item.review_status}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
            cells.append(image)
        while len(cells)<columns*4: cells.append(np.zeros((cell_size[1],cell_size[0],3),np.uint8))
        sheet=cv2.vconcat([cv2.hconcat(cells[i:i+columns]) for i in range(0,columns*4,columns)]); path=output_dir/f"contact_{start//(columns*4)+1:03d}.jpg"; cv2.imwrite(str(path),sheet); outputs.append(path)
    return outputs

def write_temporal_context(items: list[FrameAnnotation], video: Path, output_dir: Path,
                           *, split: str, delta_s: float = 0.20,
                           frame_ids: set[str] | None = None) -> list[Path]:
    """Write previous/current/next strips for difficult overlap review.

    Only development splits are accepted so this utility cannot accidentally
    reveal sealed-test context during tuning.
    """
    if split not in {"train", "validation"}:
        raise ValueError("Temporal review context is limited to train and validation")
    selected = [item for item in items if item.split == split and (not frame_ids or item.frame_id in frame_ids)]
    presentation_timestamps = video_frame_timestamps(video)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for item in selected:
        frames = []
        for label, requested in (("previous", max(0.0, item.timestamp-delta_s)),
                                 ("current", item.timestamp), ("next", item.timestamp+delta_s)):
            insertion = bisect.bisect_left(presentation_timestamps, requested)
            choices = [index for index in (insertion-1,insertion) if 0 <= index < len(presentation_timestamps)]
            frame_index = min(choices,key=lambda index:abs(presentation_timestamps[index]-requested))
            timestamp = presentation_timestamps[frame_index]
            frame = cv2.resize(read_video_frame(video, frame_index=frame_index), (480, 270), interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (0, 0), (480, 28), (0, 0, 0), -1)
            cv2.putText(frame, f"{label} t={timestamp:.3f}", (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(frame)
        strip = cv2.hconcat(frames)
        path = output_dir / f"{item.frame_id}_context.jpg"
        if not cv2.imwrite(str(path), strip, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Could not write {path}")
        outputs.append(path)
    return outputs
