from pathlib import Path
import cv2
import numpy as np
import pytest
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation, read_jsonl, write_jsonl
from monster_dataset.validation import report
from monster_dataset.annotation_io import export_yolo, write_jsonl_with_backup
from monster_dataset.contact_sheet import write_temporal_context
from monster_dataset.review_app import (
    display_box_to_source, source_box_to_display, MonsterReviewApp, ReviewSession, Viewport,
)
from monster_dataset.prelabel import detect_dynamic_ui_panels, prelabel_annotations


class FakeProposalGenerator:
    mechanism = "test_visual_proposals"

    def propose(self, image, timestamp):
        assert image.shape[:2] == (540, 960)
        return [MonsterAnnotation([100, 200, 180, 300], [140, 300],
                                  annotation_confidence=.8, review_required=True,
                                  source="codex_prelabel")]

def test_canonical_annotation_round_trip(tmp_path: Path):
    item=FrameAnnotation("f",1.25,"frame.jpg",[MonsterAnnotation([10,20,30,50],[20,50],occluded=True,visibility=.5)],category="heavy_effects")
    path=tmp_path/"annotations.jsonl"; write_jsonl([item],path); loaded=read_jsonl(path)
    assert loaded[0].monsters[0].ground_position == [20,50]
    assert loaded[0].monsters[0].occluded is True


def test_atomic_annotation_write_retains_one_previous_backup(tmp_path: Path):
    path = tmp_path / "annotations.jsonl"
    original = FrameAnnotation("old", 0, "old.jpg")
    replacement = FrameAnnotation("new", 1, "new.jpg")
    write_jsonl([original], path)
    write_jsonl_with_backup([replacement], path)
    assert read_jsonl(path)[0].frame_id == "new"
    assert read_jsonl(path.with_suffix(".jsonl.bak"))[0].frame_id == "old"

def test_pending_annotations_are_not_exported_as_training_labels(tmp_path: Path):
    item=FrameAnnotation("f",0,"missing.jpg",review_status="pending")
    with pytest.raises(ValueError, match="Refusing train export"):
        export_yolo([item],tmp_path/"labels",split="train")

def test_export_is_split_aware_and_sealed_test_is_protected(tmp_path: Path):
    image=tmp_path/"frame.jpg"; cv2.imwrite(str(image),np.zeros((540,960,3),np.uint8))
    reviewed=FrameAnnotation("f",0,str(image),[MonsterAnnotation([10,20,30,50],[20,50])],split="train",review_status="reviewed")
    result=export_yolo([reviewed],tmp_path/"yolo",split="train")
    assert result["frames"]==1 and result["instances"]==1
    with pytest.raises(ValueError,match="sealed test"):
        export_yolo([reviewed],tmp_path/"yolo",split="test")

def test_malformed_annotation_blocks_export(tmp_path: Path):
    image=tmp_path/"frame.jpg"; cv2.imwrite(str(image),np.zeros((540,960,3),np.uint8))
    malformed=FrameAnnotation("f",0,str(image),[MonsterAnnotation([10,20,30,50],[100,100])],review_status="reviewed")
    with pytest.raises(ValueError,match="ground_position"):
        export_yolo([malformed],tmp_path/"yolo",split="train")

def test_report_exposes_review_and_overlap_statistics():
    items=[FrameAnnotation("a",0,"a.jpg",[MonsterAnnotation([0,0,20,20],[10,20]),MonsterAnnotation([5,0,25,20],[15,20])],review_status="reviewed"),FrameAnnotation("b",1,"b.jpg")]
    result=report(items)
    assert result["reviewed_frames"]==1
    assert result["monster_instances"]==2
    assert result["count_distribution"]=={"0":1,"2":1}

def test_temporal_context_cannot_reveal_sealed_test(tmp_path: Path):
    item=FrameAnnotation("sealed",10,"sealed.jpg",split="test")
    with pytest.raises(ValueError,match="train and validation"):
        write_temporal_context([item],tmp_path/"missing.mp4",tmp_path/"out",split="test")

def test_review_coordinate_transform_and_sealed_guard(tmp_path: Path):
    assert display_box_to_source((10,280),(110,380))==[20.,20.,220.,220.]
    with pytest.raises(ValueError,match="train and validation"):
        MonsterReviewApp(tmp_path/"missing.jsonl",tmp_path/"missing.mp4",split="test")


def test_prelabels_remain_pending_and_record_source(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((540, 960, 3), np.uint8))
    path = tmp_path / "annotations.jsonl"
    items = [FrameAnnotation("f", 1.0, str(image), split="train")]
    write_jsonl(items, path)
    result = prelabel_annotations(items, path, split="train", generator=FakeProposalGenerator())
    loaded = read_jsonl(path)[0]
    assert loaded.review_status == "pending"
    assert loaded.monsters[0].source == "codex_prelabel"
    assert loaded.monsters[0].review_required is True
    assert result["monster_proposals"] == 1
    with pytest.raises(ValueError, match="not training-ready"):
        export_yolo([loaded], tmp_path / "yolo", split="train")


def test_prelabel_refuses_test_without_opening_image(tmp_path: Path):
    for split in ("validation", "test"):
        items = [FrameAnnotation("sealed", 1.0, "must-not-be-opened.jpg", split=split)]
        with pytest.raises(ValueError, match="test is sealed"):
            prelabel_annotations(items, tmp_path / "annotations.jsonl", split=split,
                                 generator=FakeProposalGenerator())


def test_prelabel_does_not_overwrite_reviewed_or_existing_pending_work(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((540, 960, 3), np.uint8))
    reviewed_monster = MonsterAnnotation([1, 2, 30, 40], [15, 40])
    pending_monster = MonsterAnnotation([50, 60, 90, 100], [70, 100], source="visual_review")
    items = [
        FrameAnnotation("reviewed", 1.0, str(image), [reviewed_monster], split="train", review_status="reviewed"),
        FrameAnnotation("pending-human", 2.0, str(image), [pending_monster], split="train"),
    ]
    path = tmp_path / "annotations.jsonl"
    write_jsonl(items, path)
    result = prelabel_annotations(items, path, split="train", generator=FakeProposalGenerator())
    loaded = read_jsonl(path)
    assert loaded[0].monsters[0].bbox_xyxy == [1, 2, 30, 40]
    assert loaded[1].monsters[0].source == "visual_review"
    assert result["frames_processed"] == 0
    assert result["frames_skipped_reviewed_or_non_pending"] == 1
    assert result["frames_skipped_existing_pending_annotations"] == 1


def test_replace_pending_still_never_overwrites_reviewed_labels(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((540, 960, 3), np.uint8))
    original = MonsterAnnotation([1, 2, 30, 40], [15, 40], source="visual_review")
    items = [FrameAnnotation("reviewed", 1.0, str(image), [original],
                             split="train", review_status="reviewed")]
    path = tmp_path / "annotations.jsonl"
    write_jsonl(items, path)
    prelabel_annotations(items, path, split="train", generator=FakeProposalGenerator(),
                         replace_pending=True)
    loaded = read_jsonl(path)[0]
    assert loaded.review_status == "reviewed"
    assert loaded.monsters[0].bbox_xyxy == [1, 2, 30, 40]


def test_viewport_round_trip_survives_zoom_and_pan():
    viewport = Viewport(rect=(40, 100, 1000, 600))
    source = (712.5, 431.25)
    display = viewport.source_to_display(source)
    assert viewport.display_to_source(display) == pytest.approx(source)
    viewport.zoom_at(display, 2.5)
    assert viewport.source_to_display(source) == pytest.approx(display)
    viewport.pan_display(83, -47)
    assert viewport.display_to_source(viewport.source_to_display(source)) == pytest.approx(source)
    box = [100, 200, 500, 700]
    display_box = source_box_to_display(box, viewport)
    assert display_box_to_source((display_box[0], display_box[1]),
                                 (display_box[2], display_box[3]), viewport) == pytest.approx(box)


def test_review_session_editor_operations_and_undo_redo():
    proposal = MonsterAnnotation([100, 100, 200, 240], [150, 240],
                                 source="codex_prelabel", review_required=True,
                                 annotation_confidence=.7)
    session = ReviewSession([FrameAnnotation("f", 1.0, "f.jpg", [proposal])], split="train")
    assert session.select_at((150, 150)) == 0
    session.move_selected(20, 10)
    assert session.current.monsters[0].bbox_xyxy == [120, 110, 220, 250]
    assert session.current.monsters[0].source == "human_corrected_prelabel"
    session.resize_selected("bottom_right", (260, 300))
    assert session.current.monsters[0].bbox_xyxy == [120, 110, 260, 300]
    assert session.current.monsters[0].ground_position == [190, 300]
    session.set_ground((180, 280))
    session.toggle_occluded()
    session.toggle_review_required()
    session.set_visibility(.5)
    added = session.add_box([300, 300, 400, 450])
    assert added == 1 and len(session.current.monsters) == 2
    assert session.undo() is True
    assert len(session.current.monsters) == 1
    assert session.redo() is True
    assert len(session.current.monsters) == 2
    session.delete_selected()  # undo/redo restore clears selection, so select explicitly
    session.select_at((350, 350))
    assert session.delete_selected() is True
    assert len(session.current.monsters) == 1


def test_review_session_confirm_needs_review_and_clear():
    monster = MonsterAnnotation([10, 10, 50, 80], [30, 80], source="codex_prelabel", review_required=True)
    session = ReviewSession([FrameAnnotation("f", 0, "f.jpg", [monster])], split="train")
    session.confirm()
    assert session.current.review_status == "reviewed"
    assert session.current.monsters[0].review_required is False
    assert session.current.monsters[0].source == "human_confirmed_prelabel"
    assert session.undo() is True
    assert session.current.review_status == "pending"
    session.mark_needs_review()
    assert session.current.review_status == "needs_review"
    session.clear()
    assert session.current.monsters == []


def test_review_queues_filter_without_changing_canonical_items():
    items = [
        FrameAnnotation("p", 0, "p.jpg", split="train", review_status="pending"),
        FrameAnnotation("n", 1, "n.jpg", split="train", review_status="needs_review"),
        FrameAnnotation("r", 2, "r.jpg", split="train", review_status="reviewed"),
    ]
    assert [item.frame_id for item in ReviewSession(items, split="train", queue="pending").items] == ["p"]
    assert [item.frame_id for item in ReviewSession(items, split="train", queue="needs_review").items] == ["n"]


def test_dynamic_inventory_panel_detection_masks_paired_tall_borders():
    image = np.zeros((540, 960, 3), np.uint8)
    cv2.line(image, (640, 90), (640, 510), (255, 255, 255), 3)
    cv2.line(image, (815, 100), (815, 500), (255, 255, 255), 3)
    panels = detect_dynamic_ui_panels(image)
    assert len(panels) == 1
    left, top, right, bottom = panels[0]
    assert left < 1300 < right and top < 300 < bottom
    left_side = np.zeros((540, 960, 3), np.uint8)
    cv2.line(left_side, (520, 0), (520, 500), (255, 255, 255), 3)
    cv2.line(left_side, (650, 0), (650, 500), (255, 255, 255), 3)
    assert detect_dynamic_ui_panels(left_side) == []
