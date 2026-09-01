"""Resizable OpenCV monster annotation editor with VFR temporal context."""
from __future__ import annotations

import bisect
import copy
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from perception.core import read_video_frame, video_frame_timestamps

from .annotation_io import write_jsonl_with_backup
from .prelabel import resolve_image_path
from .schema import FrameAnnotation, MonsterAnnotation, read_jsonl

SOURCE_SIZE = (1920, 1080)
CURRENT_ORIGIN = (0, 270)
CURRENT_SIZE = (960, 540)
VALID_QUEUES = {"all", "pending", "needs_review", "proposal_review_required"}
HANDLE_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass
class Viewport:
    """Lossless source/display transform; zoom and pan never touch annotations."""

    source_size: tuple[int, int] = SOURCE_SIZE
    rect: tuple[int, int, int, int] = (0, 270, 960, 540)
    zoom: float = 1.0
    center: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.center is None:
            self.center = (self.source_size[0] / 2.0, self.source_size[1] / 2.0)

    @property
    def scale(self) -> float:
        return min(self.rect[2] / self.source_size[0], self.rect[3] / self.source_size[1]) * self.zoom

    @property
    def offset(self) -> tuple[float, float]:
        assert self.center is not None
        return (self.rect[0] + self.rect[2] / 2.0 - self.center[0] * self.scale,
                self.rect[1] + self.rect[3] / 2.0 - self.center[1] * self.scale)

    def set_rect(self, rect: tuple[int, int, int, int]) -> None:
        self.rect = rect

    def fit(self) -> None:
        self.zoom = 1.0
        self.center = (self.source_size[0] / 2.0, self.source_size[1] / 2.0)

    def contains_display(self, point: tuple[float, float]) -> bool:
        x, y, width, height = self.rect
        return x <= point[0] < x + width and y <= point[1] < y + height

    def source_to_display(self, point: tuple[float, float]) -> tuple[float, float]:
        ox, oy = self.offset
        return ox + point[0] * self.scale, oy + point[1] * self.scale

    def display_to_source(self, point: tuple[float, float], *, clamp: bool = True) -> tuple[float, float]:
        ox, oy = self.offset
        result = ((point[0] - ox) / self.scale, (point[1] - oy) / self.scale)
        if not clamp:
            return result
        return (max(0.0, min(float(self.source_size[0]), result[0])),
                max(0.0, min(float(self.source_size[1]), result[1])))

    def pan_display(self, dx: float, dy: float) -> None:
        assert self.center is not None
        self.center = (self.center[0] - dx / self.scale, self.center[1] - dy / self.scale)

    def zoom_at(self, display_point: tuple[float, float], factor: float) -> None:
        anchor = self.display_to_source(display_point, clamp=False)
        old_display = self.source_to_display(anchor)
        self.zoom = max(1.0, min(12.0, self.zoom * factor))
        new_display = self.source_to_display(anchor)
        self.pan_display(old_display[0] - new_display[0], old_display[1] - new_display[1])


def display_box_to_source(start: tuple[int, int], end: tuple[int, int],
                          viewport: Viewport | None = None) -> list[float]:
    view = viewport or Viewport()
    first, second = view.display_to_source(start), view.display_to_source(end)
    return [min(first[0], second[0]), min(first[1], second[1]),
            max(first[0], second[0]), max(first[1], second[1])]


def source_box_to_display(box: list[float], viewport: Viewport) -> list[float]:
    first = viewport.source_to_display((box[0], box[1]))
    second = viewport.source_to_display((box[2], box[3]))
    return [first[0], first[1], second[0], second[1]]


class ReviewSession:
    """Rendering-independent annotation state, including per-frame undo/redo."""

    def __init__(self, items: list[FrameAnnotation], *, split: str, queue: str = "all",
                 start_id: str | None = None):
        if split not in {"train", "validation"}:
            raise ValueError("Interactive review is limited to train and validation")
        if queue not in VALID_QUEUES:
            raise ValueError(f"queue must be one of {sorted(VALID_QUEUES)}")
        self.all_items = items
        split_items = [item for item in items if item.split == split]
        predicates = {
            "all": lambda item: True,
            "pending": lambda item: item.review_status == "pending",
            "needs_review": lambda item: item.review_status == "needs_review",
            "proposal_review_required": lambda item: (
                item.review_status != "reviewed" and any(
                    monster.review_required or monster.source == "codex_prelabel" for monster in item.monsters
                )
            ),
        }
        self.items = [item for item in split_items if predicates[queue](item)]
        if not self.items:
            raise ValueError(f"No {split} annotations in queue {queue!r}")
        self.split = split
        self.queue = queue
        self.index = next((i for i, item in enumerate(self.items) if item.frame_id == start_id), 0)
        self.selected_index: int | None = None
        self.last_hit_candidates: list[int] = []
        self.undo_stacks: dict[str, list[tuple[list[MonsterAnnotation], str]]] = {}
        self.redo_stacks: dict[str, list[tuple[list[MonsterAnnotation], str]]] = {}
        self.dirty = False

    @property
    def current(self) -> FrameAnnotation:
        return self.items[self.index]

    def _snapshot(self) -> tuple[list[MonsterAnnotation], str]:
        return copy.deepcopy(self.current.monsters), self.current.review_status

    def _restore(self, state: tuple[list[MonsterAnnotation], str]) -> None:
        self.current.monsters, self.current.review_status = copy.deepcopy(state[0]), state[1]
        self.selected_index = None
        self.dirty = True

    def _record(self) -> None:
        frame_id = self.current.frame_id
        self.undo_stacks.setdefault(frame_id, []).append(self._snapshot())
        self.redo_stacks[frame_id] = []
        self.dirty = True

    def undo(self) -> bool:
        stack = self.undo_stacks.setdefault(self.current.frame_id, [])
        if not stack:
            return False
        self.redo_stacks.setdefault(self.current.frame_id, []).append(self._snapshot())
        self._restore(stack.pop())
        return True

    def redo(self) -> bool:
        stack = self.redo_stacks.setdefault(self.current.frame_id, [])
        if not stack:
            return False
        self.undo_stacks.setdefault(self.current.frame_id, []).append(self._snapshot())
        self._restore(stack.pop())
        return True

    @staticmethod
    def _valid_box(box: list[float]) -> list[float]:
        x1, x2 = sorted((max(0.0, min(SOURCE_SIZE[0], box[0])), max(0.0, min(SOURCE_SIZE[0], box[2]))))
        y1, y2 = sorted((max(0.0, min(SOURCE_SIZE[1], box[1])), max(0.0, min(SOURCE_SIZE[1], box[3]))))
        if x2 - x1 < 4.0 or y2 - y1 < 4.0:
            raise ValueError("Monster box must be at least 4 source pixels in each dimension")
        return [x1, y1, x2, y2]

    @staticmethod
    def _mark_corrected(monster: MonsterAnnotation) -> None:
        if monster.source in {"codex_prelabel", "human_confirmed_prelabel"}:
            monster.source = "human_corrected_prelabel"

    def add_box(self, box: list[float]) -> int:
        box = self._valid_box(box)
        self._record()
        self.current.monsters.append(MonsterAnnotation(
            box, [(box[0] + box[2]) / 2.0, box[3]], source="visual_review",
            review_required=False, annotation_confidence=1.0,
        ))
        self.selected_index = len(self.current.monsters) - 1
        return self.selected_index

    def select_at(self, point: tuple[float, float], *, cycle: bool = False) -> int | None:
        candidates = [index for index, monster in enumerate(self.current.monsters)
                      if monster.bbox_xyxy[0] <= point[0] <= monster.bbox_xyxy[2]
                      and monster.bbox_xyxy[1] <= point[1] <= monster.bbox_xyxy[3]]
        candidates.sort(key=lambda index: ((self.current.monsters[index].bbox_xyxy[2] - self.current.monsters[index].bbox_xyxy[0]) *
                                           (self.current.monsters[index].bbox_xyxy[3] - self.current.monsters[index].bbox_xyxy[1])))
        if not candidates:
            self.selected_index = None
            self.last_hit_candidates = []
            return None
        if (cycle or candidates == self.last_hit_candidates) and self.selected_index in candidates:
            position = (candidates.index(self.selected_index) + 1) % len(candidates)
            self.selected_index = candidates[position]
        else:
            self.selected_index = candidates[0]
        self.last_hit_candidates = candidates
        return self.selected_index

    def cycle_selection(self) -> int | None:
        candidates = self.last_hit_candidates or list(range(len(self.current.monsters)))
        if not candidates:
            self.selected_index = None
            return None
        if self.selected_index not in candidates:
            self.selected_index = candidates[0]
        else:
            self.selected_index = candidates[(candidates.index(self.selected_index) + 1) % len(candidates)]
        return self.selected_index

    def move_selected(self, dx: float, dy: float) -> None:
        if self.selected_index is None:
            return
        monster = self.current.monsters[self.selected_index]
        width = monster.bbox_xyxy[2] - monster.bbox_xyxy[0]
        height = monster.bbox_xyxy[3] - monster.bbox_xyxy[1]
        x1 = max(0.0, min(SOURCE_SIZE[0] - width, monster.bbox_xyxy[0] + dx))
        y1 = max(0.0, min(SOURCE_SIZE[1] - height, monster.bbox_xyxy[1] + dy))
        actual_dx, actual_dy = x1 - monster.bbox_xyxy[0], y1 - monster.bbox_xyxy[1]
        self._record()
        monster.bbox_xyxy = [x1, y1, x1 + width, y1 + height]
        monster.ground_position = [monster.ground_position[0] + actual_dx, monster.ground_position[1] + actual_dy]
        self._mark_corrected(monster)

    def resize_selected(self, handle: str, point: tuple[float, float]) -> None:
        if self.selected_index is None or handle not in HANDLE_NAMES:
            return
        monster = self.current.monsters[self.selected_index]
        box = list(monster.bbox_xyxy)
        if "left" in handle:
            box[0] = point[0]
        else:
            box[2] = point[0]
        if "top" in handle:
            box[1] = point[1]
        else:
            box[3] = point[1]
        box = self._valid_box(box)
        self._record()
        monster.bbox_xyxy = box
        monster.ground_position = [(box[0] + box[2]) / 2.0, box[3]]
        self._mark_corrected(monster)

    def delete_selected(self) -> bool:
        if self.selected_index is None:
            return False
        self._record()
        self.current.monsters.pop(self.selected_index)
        self.selected_index = None
        return True

    def clear(self) -> None:
        if self.current.monsters:
            self._record()
            self.current.monsters.clear()
            self.selected_index = None

    def set_ground(self, point: tuple[float, float]) -> None:
        if self.selected_index is None:
            return
        monster = self.current.monsters[self.selected_index]
        x1, y1, x2, y2 = monster.bbox_xyxy
        self._record()
        monster.ground_position = [max(x1, min(x2, point[0])), max(y1, min(y2, point[1]))]
        self._mark_corrected(monster)

    def toggle_occluded(self) -> None:
        if self.selected_index is not None:
            self._record()
            monster = self.current.monsters[self.selected_index]
            monster.occluded = not monster.occluded
            self._mark_corrected(monster)

    def toggle_review_required(self) -> None:
        if self.selected_index is not None:
            self._record()
            monster = self.current.monsters[self.selected_index]
            monster.review_required = not monster.review_required
            self._mark_corrected(monster)

    def set_visibility(self, value: float) -> None:
        if self.selected_index is not None:
            self._record()
            monster = self.current.monsters[self.selected_index]
            monster.visibility = value
            self._mark_corrected(monster)

    def confirm(self) -> None:
        self._record()
        self.current.review_status = "reviewed"
        for monster in self.current.monsters:
            if monster.source == "codex_prelabel":
                monster.source = "human_confirmed_prelabel"
            monster.review_required = False

    def mark_needs_review(self) -> None:
        self._record()
        self.current.review_status = "needs_review"

    def navigate(self, offset: int) -> bool:
        target = max(0, min(len(self.items) - 1, self.index + offset))
        if target == self.index:
            return False
        self.index = target
        self.selected_index = None
        self.last_hit_candidates = []
        return True

    def progress(self) -> dict[str, int]:
        split_items = [item for item in self.all_items if item.split == self.split]
        return {
            "reviewed": sum(item.review_status == "reviewed" for item in split_items),
            "pending": sum(item.review_status == "pending" for item in split_items),
            "needs_review": sum(item.review_status == "needs_review" for item in split_items),
        }


class MonsterReviewApp:
    def __init__(self, annotations: Path, video: Path, *, split: str, start_id: str | None = None,
                 delta_s: float = .2, queue: str = "all"):
        if split not in {"train", "validation"}:
            raise ValueError("Interactive review is limited to train and validation")
        self.annotations_path = annotations
        self.video = video
        self.delta_s = delta_s
        self.session = ReviewSession(read_jsonl(annotations), split=split, queue=queue, start_id=start_id)
        self.timestamps = video_frame_timestamps(video)
        self.viewport = Viewport()
        self.window = "Monster review"
        self.frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.image_cache: dict[str, np.ndarray] = {}
        self.interaction: dict | None = None
        self.canvas_size = (1440, 900)

    def _frame_index(self, timestamp: float) -> int:
        insertion = bisect.bisect_left(self.timestamps, timestamp)
        choices = [i for i in (insertion - 1, insertion) if 0 <= i < len(self.timestamps)]
        return min(choices, key=lambda i: abs(self.timestamps[i] - timestamp))

    def _frame(self, timestamp: float) -> np.ndarray:
        index = self._frame_index(timestamp)
        if index not in self.frame_cache:
            self.frame_cache[index] = read_video_frame(self.video, frame_index=index)
            while len(self.frame_cache) > 8:
                self.frame_cache.popitem(last=False)
        return self.frame_cache[index]

    def _current_image(self) -> np.ndarray:
        item = self.session.current
        if item.frame_id not in self.image_cache:
            path = resolve_image_path(item.image_path, self.annotations_path)
            image = cv2.imread(str(path))
            if image is None:
                image = self._frame(item.timestamp)
            self.image_cache = {item.frame_id: image}
        return self.image_cache[item.frame_id]

    def _layout(self) -> tuple[int, int, int, int]:
        width, height = self.canvas_size
        top = max(110, min(180, round(height * .18)))
        status = max(80, min(115, round(height * .12)))
        controls = max(300, min(380, round(width * .24)))
        main = (0, top, width - controls, height - top - status)
        self.viewport.set_rect(main)
        return top, status, controls, main[2]

    @staticmethod
    def _put(canvas: np.ndarray, text: str, point: tuple[int, int], scale: float = .48,
             color: tuple[int, int, int] = (225, 225, 225), thickness: int = 1) -> None:
        cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _draw_thumbnail(self, canvas: np.ndarray, frame: np.ndarray, rect: tuple[int, int, int, int], label: str) -> None:
        x, y, width, height = rect
        scale = min(width / frame.shape[1], height / frame.shape[0])
        resized = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
        left, top = x + (width - resized.shape[1]) // 2, y + (height - resized.shape[0]) // 2
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (90, 90, 90), 1)
        cv2.rectangle(canvas, (x, y), (x + width, y + 22), (0, 0, 0), -1)
        self._put(canvas, label, (x + 7, y + 16), .43)

    def _draw_main_image(self, canvas: np.ndarray, image: np.ndarray) -> None:
        ox, oy = self.viewport.offset
        image_to_source_x = SOURCE_SIZE[0] / image.shape[1]
        image_to_source_y = SOURCE_SIZE[1] / image.shape[0]
        matrix = np.asarray([[self.viewport.scale * image_to_source_x, 0.0, ox],
                             [0.0, self.viewport.scale * image_to_source_y, oy]], dtype=np.float32)
        warped = cv2.warpAffine(image, matrix, self.canvas_size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(12, 12, 12))
        x, y, width, height = self.viewport.rect
        canvas[y:y + height, x:x + width] = warped[y:y + height, x:x + width]

    @staticmethod
    def _dashed_rectangle(canvas: np.ndarray, box: list[int], color: tuple[int, int, int], thickness: int = 2) -> None:
        x1, y1, x2, y2 = box
        for start in range(x1, x2, 12):
            cv2.line(canvas, (start, y1), (min(start + 7, x2), y1), color, thickness)
            cv2.line(canvas, (start, y2), (min(start + 7, x2), y2), color, thickness)
        for start in range(y1, y2, 12):
            cv2.line(canvas, (x1, start), (x1, min(start + 7, y2)), color, thickness)
            cv2.line(canvas, (x2, start), (x2, min(start + 7, y2)), color, thickness)

    def _box_color(self, monster: MonsterAnnotation, selected: bool) -> tuple[int, int, int]:
        if selected:
            return (255, 255, 255)
        if monster.source == "codex_prelabel":
            return (255, 80, 210)
        if monster.source == "human_corrected_prelabel":
            return (0, 180, 255)
        return (70, 240, 70)

    def _handles(self, index: int) -> dict[str, tuple[float, float]]:
        box = source_box_to_display(self.session.current.monsters[index].bbox_xyxy, self.viewport)
        return {"top_left": (box[0], box[1]), "top_right": (box[2], box[1]),
                "bottom_left": (box[0], box[3]), "bottom_right": (box[2], box[3])}

    def _hit_handle(self, point: tuple[int, int]) -> str | None:
        if self.session.selected_index is None:
            return None
        for name, handle in self._handles(self.session.selected_index).items():
            if abs(point[0] - handle[0]) <= 9 and abs(point[1] - handle[1]) <= 9:
                return name
        return None

    def _draw_annotations(self, canvas: np.ndarray) -> None:
        for index, monster in enumerate(self.session.current.monsters):
            selected = index == self.session.selected_index
            box = [round(value) for value in source_box_to_display(monster.bbox_xyxy, self.viewport)]
            color = self._box_color(monster, selected)
            if monster.review_required:
                self._dashed_rectangle(canvas, box, color, 3 if selected else 2)
            else:
                cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 3 if selected else 2)
            ground = self.viewport.source_to_display(tuple(monster.ground_position))
            cv2.drawMarker(canvas, (round(ground[0]), round(ground[1])), color, cv2.MARKER_CROSS, 12, 2)
            tags = []
            if monster.source == "codex_prelabel":
                tags.append(f"proposal {monster.annotation_confidence:.2f}")
            elif monster.source == "human_corrected_prelabel":
                tags.append("corrected")
            else:
                tags.append("human")
            if monster.occluded:
                tags.append("OCCLUDED")
            if monster.review_required:
                tags.append("REVIEW")
            label = (f"M{index + 1} {' '.join(tags)}" if selected
                     else f"M{index + 1} {'P' + format(monster.annotation_confidence, '.2f') if monster.source == 'codex_prelabel' else tags[0]}")
            self._put(canvas, label, (box[0], max(self.viewport.rect[1] + 16, box[1] - 5)),
                      .42, color, 1)
            if selected:
                for handle in self._handles(index).values():
                    cv2.rectangle(canvas, (round(handle[0]) - 5, round(handle[1]) - 5),
                                  (round(handle[0]) + 5, round(handle[1]) + 5), color, -1)
        if self.interaction and self.interaction["kind"] == "add":
            preview = display_box_to_source(self.interaction["start"], self.interaction["current"], self.viewport)
            box = [round(value) for value in source_box_to_display(preview, self.viewport)]
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 1)

    def _canvas(self) -> np.ndarray:
        width, height = self.canvas_size
        canvas = np.full((height, width, 3), 22, np.uint8)
        top, status_height, _controls_width, main_width = self._layout()
        item = self.session.current
        thumb_width = width // 3
        contexts = (("PREVIOUS", max(0.0, item.timestamp - self.delta_s)),
                    ("CURRENT CONTEXT", item.timestamp), ("NEXT", item.timestamp + self.delta_s))
        for index, (label, timestamp) in enumerate(contexts):
            self._draw_thumbnail(canvas, self._frame(timestamp), (index * thumb_width, 0,
                                 thumb_width if index < 2 else width - index * thumb_width, top), label)
        self._draw_main_image(canvas, self._current_image())
        self._draw_annotations(canvas)
        controls_x = main_width + 12
        progress = self.session.progress()
        lines = [
            f"{self.session.split.upper()}  {self.session.queue}",
            f"frame {self.session.index + 1} / {len(self.session.items)}",
            f"{item.frame_id}  t={item.timestamp:.3f}",
            f"status: {item.review_status.upper()}",
            f"reviewed {progress['reviewed']}  pending {progress['pending']}",
            f"needs review {progress['needs_review']}",
            f"current proposals/boxes: {len(item.monsters)}", "",
            "Enter/R confirm + next   E needs review",
            "A/Left previous   D/Right next",
            "drag empty: add   drag box: move",
            "corner handles: resize   Del: delete",
            "Shift+click: ground point   Tab: cycle",
            "O: occluded   U: review flag",
            "1..5: visibility   0: clear boxes",
            "wheel: zoom   middle drag: pan   F: fit",
            "Ctrl+Z/Y: undo/redo   S: save",
            "Q/Esc: save and quit",
        ]
        for row, line in enumerate(lines):
            self._put(canvas, line, (controls_x, top + 27 + row * 27), .44,
                      (100, 220, 255) if row == 3 else (225, 225, 225))
        status_y = height - status_height
        cv2.rectangle(canvas, (0, status_y), (width, height), (12, 12, 12), -1)
        selected = "none" if self.session.selected_index is None else f"M{self.session.selected_index + 1}"
        self._put(canvas, f"Selected: {selected}    zoom: {self.viewport.zoom:.2f}x    autosave: on (single .bak)",
                  (12, status_y + 28), .5)
        self._put(canvas, "Pink dashed = pending proposal | orange = human-corrected | green = human-created | white = selected",
                  (12, status_y + 58), .46, (190, 190, 190))
        return canvas

    def _mouse(self, event: int, x: int, y: int, flags: int, _param: object) -> None:
        point = (x, y)
        if event == cv2.EVENT_MOUSEWHEEL and self.viewport.contains_display(point):
            self.viewport.zoom_at(point, 1.25 if flags > 0 else .8)
            return
        if event == cv2.EVENT_MBUTTONDOWN and self.viewport.contains_display(point):
            self.interaction = {"kind": "pan", "current": point}
            return
        if event == cv2.EVENT_MOUSEMOVE and self.interaction:
            if self.interaction["kind"] == "pan":
                previous = self.interaction["current"]
                self.viewport.pan_display(x - previous[0], y - previous[1])
            self.interaction["current"] = point
            return
        if event == cv2.EVENT_MBUTTONUP and self.interaction and self.interaction["kind"] == "pan":
            self.interaction = None
            return
        if event == cv2.EVENT_RBUTTONDOWN and self.viewport.contains_display(point):
            source = self.viewport.display_to_source(point)
            self.session.select_at(source)
            self.session.delete_selected()
            return
        if event == cv2.EVENT_LBUTTONDOWN and self.viewport.contains_display(point):
            source = self.viewport.display_to_source(point)
            if flags & cv2.EVENT_FLAG_SHIFTKEY and self.session.selected_index is not None:
                self.session.set_ground(source)
                return
            handle = self._hit_handle(point)
            if handle:
                self.interaction = {"kind": "resize", "start": point, "current": point, "handle": handle}
                return
            selected = self.session.select_at(source)
            self.interaction = {"kind": "move" if selected is not None else "add",
                                "start": point, "current": point, "source_start": source}
            return
        if event == cv2.EVENT_LBUTTONUP and self.interaction and self.interaction["kind"] != "pan":
            action = self.interaction
            self.interaction = None
            source_end = self.viewport.display_to_source(point)
            try:
                if action["kind"] == "add":
                    self.session.add_box(display_box_to_source(action["start"], point, self.viewport))
                elif action["kind"] == "move":
                    source_start = action["source_start"]
                    if abs(point[0] - action["start"][0]) + abs(point[1] - action["start"][1]) >= 3:
                        self.session.move_selected(source_end[0] - source_start[0], source_end[1] - source_start[1])
                elif action["kind"] == "resize":
                    self.session.resize_selected(action["handle"], source_end)
            except ValueError:
                pass

    def save(self) -> None:
        write_jsonl_with_backup(self.session.all_items, self.annotations_path)
        self.session.dirty = False

    def _advance(self) -> None:
        self.save()
        self.session.navigate(1)
        self.viewport.fit()

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, *self.canvas_size)
        cv2.setMouseCallback(self.window, self._mouse)
        while True:
            try:
                _, _, width, height = cv2.getWindowImageRect(self.window)
                if width >= 900 and height >= 600:
                    self.canvas_size = (width, height)
            except cv2.error:
                pass
            cv2.imshow(self.window, self._canvas())
            key = cv2.waitKeyEx(30)
            low = key & 0xFF
            if key == -1:
                continue
            if low in (ord("q"), 27):
                self.save()
                break
            if low == ord("s"):
                self.save()
            elif low == ord("a") or key in (81, 2424832):
                self.save(); self.session.navigate(-1); self.viewport.fit()
            elif low == ord("d") or key in (83, 2555904):
                self.save(); self.session.navigate(1); self.viewport.fit()
            elif low in (13, 10, ord("r")):
                self.session.confirm(); self._advance()
            elif low == ord("e"):
                self.session.mark_needs_review(); self._advance()
            elif key in (3014656,) or low in (8, 127):
                self.session.delete_selected()
            elif low == ord("o"):
                self.session.toggle_occluded()
            elif low == ord("u"):
                self.session.toggle_review_required()
            elif low in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                self.session.set_visibility({ord("1"): 1.0, ord("2"): .75, ord("3"): .5,
                                             ord("4"): .25, ord("5"): 0.0}[low])
            elif low == ord("0"):
                self.session.clear()
            elif low == ord("f"):
                self.viewport.fit()
            elif low == 9:
                self.session.cycle_selection()
            elif low == 26:
                self.session.undo()
            elif low == 25:
                self.session.redo()
        cv2.destroyWindow(self.window)
