"""Extract 15 original frames per extra APO source for a grouped, saved review order."""
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.core import probe_frame_timestamps, sha256_file

MAP = ROOT / "maps/APO"
STAGING = MAP / "dataset/runs/additional-sources"
SOURCES = [("APO_sample1", "apo-sample1", 20260930),
           ("APO_sample3_woSpecial", "apo-sample3", 20260931)]


def main():
    if STAGING.exists():
        raise FileExistsError("Extraction already staged; refusing to replace it")
    STAGING.mkdir(parents=True)
    sources, entries = [], []
    for name, prefix, seed in SOURCES:
        video = Path("C:/projects/input-flag-inspector/saves_m") / name / "screen.mp4"
        digest = sha256_file(video)
        timestamps = probe_frame_timestamps(video)
        indices = sorted(random.Random(seed).sample(range(len(timestamps)), 15))
        selected = set(indices)
        capture = cv2.VideoCapture(str(video))
        assert capture.isOpened()
        source_entries = []
        decoded = 0
        try:
            while capture.grab():
                index = decoded
                decoded += 1
                if index not in selected:
                    continue
                ok, frame = capture.retrieve()
                assert ok and frame.shape == (1080, 1920, 3)
                frame_id = f"{prefix}-{len(source_entries):04d}"
                destination = MAP / "dataset/images" / f"{frame_id}_{timestamps[index]:010.3f}.png"
                assert not destination.exists()
                assert cv2.imwrite(str(destination), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                assert np.array_equal(cv2.imread(str(destination)), frame)
                source_entries.append(dict(frame_id=frame_id, source_frame=index,
                    timestamp=timestamps[index], source_name=name,
                    source_video=video.as_posix(), source_video_sha256=digest,
                    image_path=destination.relative_to(ROOT).as_posix(), image_sha256=sha256_file(destination)))
        finally:
            capture.release()
        assert decoded == len(timestamps) and len(source_entries) == 15
        assert sha256_file(video) == digest
        random.Random(seed + 100).shuffle(source_entries)
        for entry in source_entries:
            entry["display_position"] = 51 + len(entries)
            entries.append(entry)
        sources.append(dict(name=name, video=video.as_posix(), video_sha256=digest,
            frame_count=decoded, sample_seed=seed, shuffle_seed=seed+100,
            display_positions=[source_entries[0]["display_position"], source_entries[-1]["display_position"]]))
    payload = dict(sources=sources, frames=entries)
    (STAGING / "extraction.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    for start in range(0,30,3):
        strips = []
        for entry in entries[start:start+3]:
            frame = cv2.imread(str(ROOT / entry["image_path"]))
            strip = frame[430:770].copy()
            for x in range(100,1900,100):
                cv2.putText(strip,str(x),(x-15,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            title = np.zeros((30,1920,3),np.uint8)
            cv2.putText(title,f'{entry["display_position"]} / {entry["frame_id"]} / {entry["timestamp"]:.3f}s',
                (8,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            strips.append(cv2.resize(cv2.vconcat([title,strip]),(1536,296)))
        assert cv2.imwrite(str(STAGING/f"source-{start//3+1:02d}.jpg"),cv2.vconcat(strips))
    print(json.dumps(sources,indent=2))


if __name__ == "__main__":
    main()
