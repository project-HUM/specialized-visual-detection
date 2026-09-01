"""Pending-only capture-local monster proposal generation for TRAIN."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from perception.core import FIXED_UI_RECTS, SessionProfile
from perception.specialized_detector import MonsterDetection, SpecializedMonsterDetector, TemplateMonsterDetector
from perception.tracking import deduplicate_detections

from .annotation_io import write_jsonl_with_backup
from .schema import FrameAnnotation, MonsterAnnotation


class ProposalGenerator(Protocol):
    mechanism: str

    def propose(self, image: np.ndarray, timestamp: float) -> list[MonsterAnnotation]: ...


def detect_dynamic_ui_panels(image: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Find tall inventory-like panels from paired long vertical borders."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=90,
                            minLineLength=160, maxLineGap=18)
    if lines is None:
        return []
    vertical: list[tuple[int, int, int]] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(int(x2) - int(x1)) <= 4 and abs(int(y2) - int(y1)) >= 160:
            vertical.append((round((int(x1) + int(x2)) / 2), min(int(y1), int(y2)), max(int(y1), int(y2))))
    panels: list[tuple[float, float, float, float]] = []
    width_scale, height_scale = 1920.0 / image.shape[1], 1080.0 / image.shape[0]
    for left_x, left_top, left_bottom in vertical:
        if left_x < image.shape[1] * .60:
            continue
        candidates = [line for line in vertical
                      if image.shape[1] * .12 <= line[0] - left_x <= image.shape[1] * .25]
        if not candidates:
            continue
        right_x, right_top, right_bottom = min(candidates, key=lambda line: line[0] - left_x)
        top, bottom = min(left_top, right_top), max(left_bottom, right_bottom)
        if bottom - top < image.shape[0] * .42:
            continue
        panel = ((left_x - 6) * width_scale, max(0, top - 18) * height_scale,
                 (right_x + 6) * width_scale, min(image.shape[0], bottom + 18) * height_scale)
        if not any(abs(panel[0] - existing[0]) < 20 for existing in panels):
            panels.append(panel)
    return panels


def resolve_image_path(image_path: str, annotations_path: Path) -> Path:
    candidate = Path(image_path)
    if candidate.is_file():
        return candidate
    repository_relative = annotations_path.resolve().parent.parent / candidate
    if repository_relative.is_file():
        return repository_relative
    raise FileNotFoundError(f"Could not read annotation image: {image_path}")


class TemplateProposalGenerator:
    """Use the existing capture-local template bank without temporal persistence."""

    mechanism = "capture_local_template_bank"

    def __init__(self, profile_path: Path):
        self.profile = SessionProfile.load(profile_path)
        templates = [cv2.imread(str(self.profile.asset(path)), cv2.IMREAD_GRAYSCALE)
                     for path in self.profile.monster_templates]
        if not templates or any(template is None for template in templates):
            raise ValueError("Session profile does not contain a readable monster template bank")
        self.detector: SpecializedMonsterDetector = TemplateMonsterDetector(
            templates, threshold=float(self.profile.thresholds.get("monster_match", 0.42))
        )

    def _scene(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = image.shape[:2]
        scale_x = width / self.profile.source_width
        scale_y = height / self.profile.source_height
        top = round(self.profile.scene_top * scale_y)
        bottom = round(self.profile.scene_bottom * scale_y)
        crop = image[max(0, top):min(height, bottom)]
        target = (
            round(self.profile.source_width * self.profile.scale),
            round((self.profile.scene_bottom - self.profile.scene_top) * self.profile.scale),
        )
        scene = cv2.resize(crop, target, interpolation=cv2.INTER_AREA)
        return scene, self.profile.scale, float(self.profile.scene_top)

    @staticmethod
    def _inside_fixed_ui(point: tuple[float, float]) -> bool:
        x, y = point
        return any(left <= x <= right and top <= y <= bottom
                   for left, top, right, bottom in FIXED_UI_RECTS)

    def propose(self, image: np.ndarray, timestamp: float) -> list[MonsterAnnotation]:
        scene, scale, scene_top = self._scene(image)
        normalized: list[dict] = []
        dynamic_ui = detect_dynamic_ui_panels(image)
        for index, detection in enumerate(self.detector.detect(scene, timestamp)):
            x1, y1, x2, y2 = detection.bbox
            box = [x1 / scale, y1 / scale + scene_top,
                   x2 / scale, y2 / scale + scene_top]
            ground = ((box[0] + box[2]) / 2.0, box[3])
            if self._inside_fixed_ui(ground) or any(
                    left <= ground[0] <= right and top <= ground[1] <= bottom
                    for left, top, right, bottom in dynamic_ui):
                continue
            normalized.append({
                "box": box,
                "score": float(detection.detector_confidence),
                "source_index": index,
            })
        retained, _ = deduplicate_detections(
            normalized,
            iou_threshold=float(self.profile.thresholds.get("monster_dedup_iou", 0.88)),
            center_fraction=float(self.profile.thresholds.get("monster_dedup_center_fraction", 0.22)),
            size_similarity=float(self.profile.thresholds.get("monster_dedup_size_similarity", 0.72)),
        )
        proposals: list[MonsterAnnotation] = []
        for item in retained:
            box = [round(max(0.0, min(float(limit), float(value))), 2)
                   for value, limit in zip(item["box"], (
                       self.profile.source_width, self.profile.source_height,
                       self.profile.source_width, self.profile.source_height,
                   ))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            confidence = max(0.0, min(1.0, float(item.get("score", 0.0))))
            proposals.append(MonsterAnnotation(
                bbox_xyxy=box,
                ground_position=[round((box[0] + box[2]) / 2.0, 2), box[3]],
                visibility=1.0,
                occluded=False,
                annotation_confidence=round(confidence, 4),
                review_required=True,
                source="codex_prelabel",
            ))
        return proposals


class Owlv2ProposalGenerator:
    """Open-vocabulary visual proposals using the locally cached OWLv2 model."""

    mechanism = "owlv2_text_pirate_monster"

    def __init__(self, *, model_name: str = "google/owlv2-base-patch16-ensemble",
                 threshold: float = .30, prompts: list[str] | None = None,
                 local_files_only: bool = True):
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:
            raise RuntimeError("OWLv2 pre-labeling requires torch, transformers, and Pillow") from exc
        self.torch = torch
        self.threshold = threshold
        self.prompts = prompts or ["pirate monster", "pirate mushroom", "cartoon pirate monster"]
        self.processor = Owlv2Processor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval()

    @staticmethod
    def _plausible_box(box: list[float], image_width: int, image_height: int) -> bool:
        width, height = box[2] - box[0], box[3] - box[1]
        source_width, source_height = width * 1920.0 / image_width, height * 1080.0 / image_height
        if not (45.0 <= source_width <= 320.0 and 45.0 <= source_height <= 280.0):
            return False
        aspect = source_width / max(source_height, 1.0)
        return .32 <= aspect <= 2.5

    @staticmethod
    def _inside_fixed_ui(point: tuple[float, float]) -> bool:
        x, y = point
        return any(left <= x <= right and top <= y <= bottom
                   for left, top, right, bottom in FIXED_UI_RECTS)

    def propose(self, image: np.ndarray, timestamp: float) -> list[MonsterAnnotation]:
        from PIL import Image

        dynamic_ui = detect_dynamic_ui_panels(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        text_labels = [self.prompts]
        inputs = self.processor(text=text_labels, images=pil_image, return_tensors="pt")
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([(image.shape[0], image.shape[1])])
        processed = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.threshold,
            text_labels=text_labels,
        )[0]
        scale_x, scale_y = 1920.0 / image.shape[1], 1080.0 / image.shape[0]
        normalized: list[dict] = []
        labels = processed.get("text_labels") or []
        for index, (box_tensor, score_tensor, label) in enumerate(zip(
                processed["boxes"], processed["scores"], labels, strict=False)):
            preview_box = [float(value) for value in box_tensor.tolist()]
            if not self._plausible_box(preview_box, image.shape[1], image.shape[0]):
                continue
            box = [preview_box[0] * scale_x, preview_box[1] * scale_y,
                   preview_box[2] * scale_x, preview_box[3] * scale_y]
            ground = ((box[0] + box[2]) / 2.0, box[3])
            if self._inside_fixed_ui(ground) or any(
                    left <= ground[0] <= right and top <= ground[1] <= bottom
                    for left, top, right, bottom in dynamic_ui):
                continue
            normalized.append({"box": box, "score": float(score_tensor.item()),
                               "source_index": index, "prompt": str(label)})
        retained, _ = deduplicate_detections(normalized, iou_threshold=.86,
                                             center_fraction=.30, size_similarity=.65)
        proposals: list[MonsterAnnotation] = []
        for item in retained:
            box = [round(max(0.0, min(float(limit), float(value))), 2)
                   for value, limit in zip(item["box"], (1920, 1080, 1920, 1080))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            confidence = max(0.0, min(1.0, float(item["score"])))
            proposals.append(MonsterAnnotation(
                bbox_xyxy=box,
                ground_position=[round((box[0] + box[2]) / 2.0, 2), box[3]],
                visibility=1.0,
                occluded=False,
                annotation_confidence=round(confidence, 4),
                review_required=True,
                source="codex_prelabel",
            ))
        return proposals


def prelabel_annotations(
    items: list[FrameAnnotation],
    annotations_path: Path,
    *,
    split: str,
    generator: ProposalGenerator,
    replace_pending: bool = False,
    report_path: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Populate pending TRAIN frames with proposals, never canonical labels."""
    if split != "train":
        raise ValueError("Automatic pre-labeling is restricted to TRAIN; validation is manual and test is sealed")
    selected = [item for item in items if item.split == "train"]
    processed = skipped_reviewed = skipped_existing = 0
    proposal_count = high = medium = low = 0
    frames_with_proposals = 0
    for item in selected:
        if item.review_status != "pending":
            skipped_reviewed += 1
            continue
        if item.monsters and not replace_pending:
            skipped_existing += 1
            continue
        image = cv2.imread(str(resolve_image_path(item.image_path, annotations_path)))
        if image is None:
            raise RuntimeError(f"Could not decode annotation image: {item.image_path}")
        proposals = generator.propose(image, item.timestamp)
        item.monsters = proposals
        item.review_status = "pending"
        processed += 1
        proposal_count += len(proposals)
        frames_with_proposals += bool(proposals)
        for monster in proposals:
            if monster.annotation_confidence >= 0.75:
                high += 1
            elif monster.annotation_confidence >= 0.50:
                medium += 1
            else:
                low += 1
        if progress is not None and (processed == 1 or processed % 10 == 0 or processed == len(selected)):
            progress(processed, len(selected))
    write_jsonl_with_backup(items, annotations_path)
    result = {
        "schema_version": 1,
        "split": "train",
        "proposal_mechanism": generator.mechanism,
        "warning": "Proposals are pending review and are not ground truth or training-ready labels.",
        "frames_available": len(selected),
        "frames_processed": processed,
        "frames_with_proposals": frames_with_proposals,
        "frames_with_zero_proposals": processed - frames_with_proposals,
        "frames_skipped_reviewed_or_non_pending": skipped_reviewed,
        "frames_skipped_existing_pending_annotations": skipped_existing,
        "monster_proposals": proposal_count,
        "confidence": {"high_gte_0_75": high, "medium_gte_0_50": medium, "low_lt_0_50": low},
        "review_required": proposal_count,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
