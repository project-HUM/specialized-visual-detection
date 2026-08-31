from __future__ import annotations
import csv, json
from pathlib import Path
from .schema import FrameAnnotation, MonsterAnnotation, read_jsonl, write_jsonl

def from_benchmark_csv(path: Path, image_root: Path) -> list[FrameAnnotation]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    result=[]
    for row in rows:
        result.append(FrameAnnotation(row["sample_id"], float(row["timestamp"]), str(image_root / f"{row['sample_id']}_{float(row['timestamp']):010.3f}.jpg"), category=row.get("category", "unspecified"), review_status=row.get("review_status", "pending"), notes=row.get("notes", "")))
    return result

def export_yolo(items: list[FrameAnnotation], output_dir: Path, *, include_pending: bool = False) -> dict[str,int]:
    output_dir.mkdir(parents=True, exist_ok=True); labels=0; skipped=0
    for item in items:
        if item.review_status != "reviewed" and not include_pending:
            skipped += 1; continue
        image = Path(item.image_path)
        # YOLO remains an export, never the canonical representation.
        try:
            import cv2
            frame = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if frame is None: raise FileNotFoundError(image)
            # Canonical boxes are in source capture coordinates even though
            # review images are downscaled to 960x540.
            width,height=1920,1080
        except Exception:
            skipped += 1; continue
        lines=[]
        for monster in item.monsters:
            x1,y1,x2,y2=monster.bbox_xyxy
            lines.append(f"0 {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
        (output_dir / f"{item.frame_id}.txt").write_text("\n".join(lines)+("\n" if lines else ""), encoding="utf-8")
        labels += 1
    return {"exported":labels,"skipped":skipped}
