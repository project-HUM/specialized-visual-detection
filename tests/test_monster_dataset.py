import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation, read_jsonl, write_jsonl
from monster_dataset.validation import report
from monster_dataset.annotation_io import export_yolo, write_jsonl_with_backup
from monster_dataset.contact_sheet import write_temporal_context
from monster_dataset.review_app import (
    BoxPreset, display_box_to_source, fixed_box_centered_at, fixed_box_from_drag,
    load_codex_review_frame_indices, load_pilot_frame_indices, load_review_settings, source_box_to_display,
    MonsterReviewApp, ReviewInstanceLock, ReviewSession, Viewport,
)
from monster_dataset.prelabel import detect_dynamic_ui_panels, prelabel_annotations


class FakeProposalGenerator:
    mechanism = "test_visual_proposals"

    def propose(self, image, timestamp):
        assert image.shape[:2] == (540, 960)
        return [MonsterAnnotation([100, 200, 180, 300], [140, 300],
                                  annotation_confidence=.8, review_required=True,
                                  source="codex_prelabel")]


def test_review_instance_lock_rejects_a_second_writer(tmp_path: Path):
    annotations = tmp_path / "annotations.jsonl"
    with ReviewInstanceLock(annotations):
        with pytest.raises(RuntimeError, match="Another monster reviewer"):
            with ReviewInstanceLock(annotations):
                pytest.fail("a second reviewer must not acquire the same annotation lock")

    with ReviewInstanceLock(annotations):
        pass


def test_review_app_only_writes_backup_for_actual_edits(tmp_path: Path, monkeypatch):
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.annotations_path = tmp_path / "annotations.jsonl"
    app.session = ReviewSession([FrameAnnotation("frame", 0, "frame.jpg")], split="train")
    writes = []
    monkeypatch.setattr(
        "monster_dataset.review_app.write_jsonl_with_backup",
        lambda items, path: writes.append((items, path)),
    )

    app.save()
    assert writes == []
    app.session.add_box([10, 20, 30, 50])
    app.save()
    assert len(writes) == 1
    assert app.session.dirty is False


def test_pilot_review_uses_manifest_frame_anchor_without_full_video_probe(
        tmp_path: Path, monkeypatch):
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl([FrameAnnotation("pilot-0000", 10.0, "frame.jpg", split="pilot")], annotations)
    manifest = tmp_path / "pilot_manifest.json"
    manifest.write_text(
        '{"schema":"specialized-visual-detection.pilot.v1",'
        '"frames":[{"frame_id":"pilot-0000","frame":300,"time_s":10.0}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "monster_dataset.review_app.video_frame_timestamps",
        lambda _video: pytest.fail("pilot review must not scan every video timestamp"),
    )
    monkeypatch.setattr(MonsterReviewApp, "_nominal_video_fps", staticmethod(lambda _video: 30.0))

    app = MonsterReviewApp(annotations, tmp_path / "large.mp4", split="pilot", queue="pending")
    item = app.session.current

    assert load_pilot_frame_indices(manifest) == {"pilot-0000": 300}
    assert app.timestamps is None
    assert app._frame_index(item.timestamp, item) == 300
    assert app._frame_index(item.timestamp - .2, item) == 294
    assert app._frame_index(item.timestamp + .2, item) == 306


def test_codex_train_review_uses_manifest_frame_anchor_without_full_video_probe(
        tmp_path: Path, monkeypatch):
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl([FrameAnnotation("codex-random-0000", 10.0, "frame.jpg", split="train")], annotations)
    manifest = tmp_path / "codex_manual_batch_v1_manifest.json"
    manifest.write_text(
        '{"schema":"specialized-visual-detection.codex-manual-review-batch.v1",'
        '"frames":[{"frame_id":"codex-random-0000","source_frame":300,"timestamp":10.0}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "monster_dataset.review_app.video_frame_timestamps",
        lambda _video: pytest.fail("anchored train review must not scan every video timestamp"),
    )
    monkeypatch.setattr(MonsterReviewApp, "_nominal_video_fps", staticmethod(lambda _video: 30.0))

    app = MonsterReviewApp(annotations, tmp_path / "large.mp4", split="train", queue="pending")
    item = app.session.current

    assert load_codex_review_frame_indices(manifest) == {"codex-random-0000": 300}
    assert app.timestamps is None
    assert app._frame_index(item.timestamp, item) == 300
    assert app._frame_index(item.timestamp - .2, item) == 294
    assert app._frame_index(item.timestamp + .2, item) == 306


def test_mixed_pilot_and_codex_train_review_combines_exact_frame_anchors(
        tmp_path: Path, monkeypatch):
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl([
        FrameAnnotation("pilot-0000", 10.0, "pilot.jpg", split="train"),
        FrameAnnotation("codex-random-0000", 20.0, "codex.jpg", split="train"),
    ], annotations)
    (tmp_path / "pilot_manifest.json").write_text(
        '{"schema":"specialized-visual-detection.pilot.v1",'
        '"frames":[{"frame_id":"pilot-0000","frame":300,"time_s":10.0}]}',
        encoding="utf-8",
    )
    (tmp_path / "codex_manual_batch_v1_manifest.json").write_text(
        '{"schema":"specialized-visual-detection.codex-manual-review-batch.v1",'
        '"frames":[{"frame_id":"codex-random-0000","source_frame":600,"timestamp":20.0}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "monster_dataset.review_app.video_frame_timestamps",
        lambda _video: pytest.fail("mixed anchored review must not scan every video timestamp"),
    )
    monkeypatch.setattr(MonsterReviewApp, "_nominal_video_fps", staticmethod(lambda _video: 30.0))

    app = MonsterReviewApp(annotations, tmp_path / "large.mp4", split="train", queue="all")

    assert app.timestamps is None
    assert app.source_frame_indices == {"pilot-0000": 300, "codex-random-0000": 600}


def test_reviewed_queue_excludes_new_pending_batch():
    reviewed = FrameAnnotation("old", 1.0, "old.jpg", split="train", review_status="reviewed")
    pending = FrameAnnotation("new", 2.0, "new.jpg", split="train", review_status="pending")

    session = ReviewSession([reviewed, pending], split="train", queue="reviewed")

    assert [item.frame_id for item in session.items] == ["old"]


def test_all_split_with_prefixes_keeps_batches_separate():
    items = [
        FrameAnnotation("pilot-0000", 1.0, "pilot.jpg", split="validation",
                        review_status="reviewed"),
        FrameAnnotation("codex-random-0000", 2.0, "batch1.jpg", split="train",
                        review_status="reviewed"),
        FrameAnnotation("codex-lich-late-0000", 3.0, "batch1-lich.jpg", split="validation",
                        review_status="reviewed"),
        FrameAnnotation("codex-sample4-random-0000", 4.0, "batch2.jpg", split="train",
                        review_status="pending"),
    ]

    original = ReviewSession(items, split="all", queue="reviewed",
                             frame_id_prefixes=("pilot-",))
    batch1 = ReviewSession(items, split="all", queue="reviewed",
                           frame_id_prefixes=("codex-random-", "codex-lich-"))
    batch2 = ReviewSession(items, split="all", queue="all",
                           frame_id_prefixes=("codex-sample4-",))

    assert [item.frame_id for item in original.items] == ["pilot-0000"]
    assert [item.frame_id for item in batch1.items] == [
        "codex-random-0000", "codex-lich-late-0000",
    ]
    assert [item.frame_id for item in batch2.items] == ["codex-sample4-random-0000"]
    assert batch2.progress() == {"reviewed": 0, "pending": 1, "needs_review": 0}

def test_canonical_annotation_round_trip(tmp_path: Path):
    item=FrameAnnotation("f",1.25,"frame.jpg",[MonsterAnnotation([10,20,30,50],[20,50],occluded=True,visibility=.5)],category="heavy_effects")
    path=tmp_path/"annotations.jsonl"; write_jsonl([item],path); loaded=read_jsonl(path)
    assert loaded[0].monsters[0].ground_position == [20,50]
    assert loaded[0].monsters[0].occluded is True


def test_partially_off_frame_monster_box_requires_one_sixteenth_visible_area():
    accepted = MonsterAnnotation([-150, 100, 10, 260], [-70, 260], box_preset="1")
    rejected = MonsterAnnotation([-151, 100, 9, 260], [-71, 260], box_preset="1")
    assert accepted.validate() == []
    assert any("at least 0.0625" in error for error in rejected.validate())


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


def test_yolo_export_maps_box_presets_to_distinct_classes(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((540, 960, 3), np.uint8))
    reviewed = FrameAnnotation(
        "f", 0, str(image),
        [MonsterAnnotation([10, 20, 30, 50], [20, 50], box_preset="1"),
         MonsterAnnotation([40, 20, 80, 70], [60, 70], box_preset="3")],
        split="train", review_status="reviewed",
    )
    output = tmp_path / "yolo"
    result = export_yolo(
        [reviewed], output, split="train",
        preset_classes={"1": "zombie", "2": "hero", "3": "lich"},
    )

    labels = (output / "labels/train/f.txt").read_text(encoding="utf-8").splitlines()
    assert labels[0].startswith("0 ")
    assert labels[1].startswith("2 ")
    assert result["classes"] == {"lich": 1, "zombie": 1}
    assert '0: "zombie"' in (output / "dataset.yaml").read_text(encoding="utf-8")


def test_multi_class_export_rejects_unmapped_boxes(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((540, 960, 3), np.uint8))
    reviewed = FrameAnnotation(
        "f", 0, str(image),
        [MonsterAnnotation([10, 20, 30, 50], [20, 50])],
        split="train", review_status="reviewed",
    )
    with pytest.raises(ValueError, match="every monster must use a mapped box preset"):
        export_yolo([reviewed], tmp_path / "yolo", split="train",
                    preset_classes={"1": "zombie"})


def test_yolo_export_clips_partially_off_frame_box(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((20, 20, 3), np.uint8))
    monster = MonsterAnnotation([-140, 100, 20, 260], [-60, 260], box_preset="1")
    reviewed = FrameAnnotation("partial", 0, str(image), [monster], split="train",
                               review_status="reviewed")
    output = tmp_path / "yolo"
    export_yolo([reviewed], output, split="train")
    values = [float(value) for value in (output / "labels/train/partial.txt").read_text().split()[1:]]
    assert values == pytest.approx([10 / 1920, 180 / 1080, 20 / 1920, 160 / 1080], abs=1e-6)


def test_dragging_preset_box_can_move_from_inside_to_partially_outside_frame():
    preset = BoxPreset("1", "Orange mob", 170, 121, 42)
    monster = MonsterAnnotation([100, 100, 270, 221], [185, 221], box_preset="1")
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.session = ReviewSession([FrameAnnotation("f", 0, "f.jpg", [monster])], split="train")
    app.session.selected_index = 0
    app.viewport = Viewport(rect=(0, 0, 960, 540))
    app.presets = {"1": preset}
    action = {
        "kind": "move", "start": (75, 75), "current": (75, 75),
        "source_start": (150, 150), "original_box": list(monster.bbox_xyxy),
    }

    partial, valid = app._move_preview(action, (10, 75))
    assert valid is True
    assert partial == [-30, 100, 140, 221]

    app.interaction = action
    app._mouse(cv2.EVENT_LBUTTONUP, 10, 75, 0, None)
    assert monster.bbox_xyxy == [-30, 100, 140, 221]
    assert monster.ground_position == [55, 221]


def test_move_preview_tracks_cursor_without_mutating_annotation_and_flags_invalid_position():
    monster = MonsterAnnotation([100, 100, 270, 221], [185, 221], box_preset="1")
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.session = ReviewSession([FrameAnnotation("f", 0, "f.jpg", [monster])], split="train")
    app.session.selected_index = 0
    app.viewport = Viewport(rect=(0, 0, 960, 540))
    app.presets = {"1": BoxPreset("1", "Orange mob", 170, 121, 42)}
    action = {
        "kind": "move", "source_start": (150, 150), "original_box": list(monster.bbox_xyxy),
    }

    preview, valid = app._move_preview(action, (100, 100))
    assert preview == [150, 150, 320, 271]
    assert valid is True
    assert monster.bbox_xyxy == [100, 100, 270, 221]

    invalid_preview, valid = app._move_preview(action, (-1000, 100))
    assert invalid_preview[2] < 0
    assert valid is False


def test_move_grab_area_extends_beyond_box_without_enlarging_resize_handles():
    monster = MonsterAnnotation([100, 100, 270, 221], [185, 221])
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.session = ReviewSession([FrameAnnotation("f", 0, "f.jpg", [monster])], split="train")
    app.session.selected_index = 0
    app.viewport = Viewport(rect=(0, 0, 960, 540))

    assert app._hit_handle((59, 59)) == "top_left"
    assert app._hit_handle((60, 60)) is None
    assert app.session.select_at((80, 150)) is None
    assert app.session.select_at(
        (80, 150), padding=12 / app.viewport.scale
    ) == 0

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


def test_review_settings_load_three_user_configurable_fixed_box_presets(tmp_path: Path):
    settings_path = tmp_path / "review_settings.json"
    settings_path.write_text(
        '{"box_presets":['
        '{"name":"Short","width":120,"height":130,"hue":25},'
        '{"name":"Tall","width":180,"height":210,"hue":210},'
        '{"name":"Wide","width":220,"height":110,"hue":310}]}'
    )
    settings = load_review_settings(settings_path)
    assert [(preset.name, preset.width, preset.height) for preset in settings.box_presets] == [
        ("Short", 120, 130), ("Tall", 180, 210), ("Wide", 220, 110),
    ]
    assert len({preset.color for preset in settings.box_presets}) == 3


def test_fixed_box_helpers_preserve_preset_size_and_drag_direction():
    preset = BoxPreset("1", "Mob", 160, 170, 40)
    assert fixed_box_from_drag((10, 20), (30, 40), preset) == [10, 20, 170, 190]
    assert fixed_box_from_drag((10, 20), (-30, -40), preset) == [-150, -150, 10, 20]
    assert fixed_box_centered_at((100, 200), preset) == [20, 115, 180, 285]


def test_preset_preview_is_red_until_enough_of_box_is_inside_frame():
    preset = BoxPreset("1", "Orange mob", 170, 121, 42)
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    assert app._preset_preview_color([-165, 100, 5, 221], preset) == (0, 0, 255)
    assert app._preset_preview_color([-159, 100, 11, 221], preset) == preset.color


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


def test_preset_resize_keeps_selected_box_bottom_left_fixed_and_persists(tmp_path: Path):
    settings_path = tmp_path / "review_settings.json"
    settings_path.write_text(
        '{"box_presets":['
        '{"name":"Zombie","width":170,"height":121,"hue":42},'
        '{"name":"Hero","width":181,"height":144,"hue":205},'
        '{"name":"Lich","width":160,"height":220,"hue":305}]}',
        encoding="utf-8",
    )
    lich = MonsterAnnotation([100, 100, 260, 320], [180, 320], box_preset="3")
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.settings_path = settings_path
    app.settings = load_review_settings(settings_path)
    app.presets = {preset.preset_id: preset for preset in app.settings.box_presets}
    app.active_preset_id = "3"
    app.dimension_status = ""
    app.session = ReviewSession([FrameAnnotation("f", 0, "f.jpg", [lich])], split="train")
    app.session.selected_index = 0

    assert app._adjust_preset_dimensions(1, 0) is True
    assert app.session.current.monsters[0].bbox_xyxy == [100, 100, 261, 320]
    assert app.session.current.monsters[0].ground_position == [180.5, 320]
    assert app._adjust_preset_dimensions(0, -1) is True
    assert app.session.current.monsters[0].bbox_xyxy == [100, 101, 261, 320]

    # Uppercase letter codes remain usable when Remote Desktop does not expose
    # the modifier state through GetAsyncKeyState.
    assert app._handle_key(ord("R"), control_pressed=False, shift_pressed=False) is True
    assert app.session.current.monsters[0].bbox_xyxy == [100, 101, 262, 320]

    persisted = json.loads(settings_path.read_text(encoding="utf-8"))
    assert persisted["box_presets"][2]["width"] == 162
    assert persisted["box_presets"][2]["height"] == 219


def test_ctrl_arrows_and_letters_nudge_selected_box_and_keep_it_inside_frame():
    monster = MonsterAnnotation([0, 100, 83, 231], [41.5, 231], box_preset="1")
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.session = ReviewSession([FrameAnnotation("f", 0, "f.jpg", [monster])], split="train")
    app.session.selected_index = 0
    app.dimension_status = ""

    assert app._handle_key(2424832, control_pressed=True, shift_pressed=False) is True
    assert monster.bbox_xyxy == [0, 100, 83, 231]
    assert app._handle_key(2555904, control_pressed=True, shift_pressed=False) is True
    assert monster.bbox_xyxy == [1, 100, 84, 231]
    assert app._handle_key(21, control_pressed=False, shift_pressed=False) is True  # Ctrl+U
    assert monster.bbox_xyxy == [1, 99, 84, 230]
    assert app._handle_key(4, control_pressed=False, shift_pressed=False) is True  # Ctrl+D
    assert monster.bbox_xyxy == [1, 100, 84, 231]

    partial = MonsterAnnotation([-10, 100, 73, 231], [31.5, 231], box_preset="1")
    app.session.current.monsters.append(partial)
    app.session.selected_index = 1
    assert app._handle_key(18, control_pressed=False, shift_pressed=False) is True  # Ctrl+R
    assert partial.bbox_xyxy == [-9, 100, 74, 231]


def test_add_mode_draws_new_box_over_existing_box_instead_of_selecting_it():
    existing = MonsterAnnotation([100, 100, 300, 300], [200, 300])
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.session = ReviewSession([FrameAnnotation("f", 0, "f.jpg", [existing])], split="train")
    app.viewport = Viewport(rect=(0, 0, 960, 540))
    app.interaction = None
    app.add_mode = True
    app.active_preset_id = None
    app.preset_rects = {}

    app._mouse(cv2.EVENT_LBUTTONDOWN, 80, 80, 0, None)
    assert app.interaction is not None and app.interaction["kind"] == "add"
    assert app.session.selected_index is None
    app._mouse(cv2.EVENT_LBUTTONUP, 120, 120, 0, None)

    assert len(app.session.current.monsters) == 2
    assert app.session.current.monsters[1].bbox_xyxy == [160, 160, 240, 240]


def test_dragging_preset_card_onto_image_creates_fixed_hued_box():
    preset1 = BoxPreset("1", "Mob type 1", 160, 170, 42)
    preset2 = BoxPreset("2", "Mob type 2", 180, 145, 205)
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.session = ReviewSession([FrameAnnotation("f", 0, "f.jpg")], split="train")
    app.viewport = Viewport(rect=(0, 0, 960, 540))
    app.interaction = None
    app.add_mode = False
    app.active_preset_id = None
    app.presets = {"1": preset1, "2": preset2}
    app.preset_rects = {"1": (1000, 100, 120, 50), "2": (1130, 100, 120, 50)}

    app._mouse(cv2.EVENT_LBUTTONDOWN, 1010, 110, 0, None)
    app._mouse(cv2.EVENT_MOUSEMOVE, 100, 100, 0, None)
    app._mouse(cv2.EVENT_LBUTTONUP, 100, 100, 0, None)

    monster = app.session.current.monsters[0]
    assert monster.bbox_xyxy == [120, 115, 280, 285]
    assert monster.box_preset == "1"
    assert app._box_color(monster, False) == preset1.color
    assert app._box_color(monster, True) == preset1.color
    assert app._box_color(MonsterAnnotation([0, 0, 10, 10], [5, 10], box_preset="2"), False) == preset2.color


def test_review_keyboard_uses_arrows_for_navigation_and_a_toggles_add_mode():
    items = [FrameAnnotation("first", 0, "first.jpg"),
             FrameAnnotation("second", 1, "second.jpg")]
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.session = ReviewSession(items, split="train")
    app.viewport = Viewport()
    app.interaction = {"kind": "add"}
    app.add_mode = False
    app.active_preset_id = None
    app.presets = {
        "1": BoxPreset("1", "Zombie", 170, 121, 42),
        "2": BoxPreset("2", "Hero", 181, 144, 205),
        "3": BoxPreset("3", "Lich", 160, 220, 305),
    }
    app.dimension_status = ""
    saved = []
    app.save = lambda: saved.append(True)

    assert app._handle_key(ord("a")) is True
    assert app.add_mode is True and app.interaction is None
    assert app.session.index == 0
    assert app._handle_key(ord("d")) is True
    assert app.session.index == 0
    assert app._handle_key(ord("1"), control_pressed=True) is True
    assert app.active_preset_id == "1" and app.add_mode is False
    assert app._handle_key(ord("3"), control_pressed=True) is True
    assert app.active_preset_id == "3" and app.add_mode is False
    assert app._handle_key(27) is True
    assert app.active_preset_id is None
    assert app._handle_key(2555904) is True
    assert app.session.index == 1
    saved.clear()

    assert app._handle_key(ord("a")) is True
    assert app.add_mode is True
    app._handle_key(ord("a"))
    assert app.add_mode is False
    app._handle_key(ord("a"))
    assert app._handle_key(27) is True
    assert app.add_mode is False
    assert app._handle_key(27) is True
    assert not saved
    assert app._handle_key(ord("q")) is False
    assert saved


def test_mouse_wheel_scrolls_help_panel_without_zooming_image():
    app = MonsterReviewApp.__new__(MonsterReviewApp)
    app.viewport = Viewport(rect=(0, 0, 900, 600))
    app.help_panel_rect = (1000, 100, 300, 400)
    app.help_scroll_lines = 0
    app.help_max_scroll_lines = 9

    app._mouse(cv2.EVENT_MOUSEWHEEL, 1100, 200, -1, None)
    assert app.help_scroll_lines == 3
    assert app.viewport.zoom == 1.0

    app._mouse(cv2.EVENT_MOUSEWHEEL, 1100, 200, 1, None)
    assert app.help_scroll_lines == 0


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


def test_proposal_review_queue_starts_at_next_work_with_reviewed_history_before_it():
    reviewed = FrameAnnotation("reviewed", 0, "reviewed.jpg", split="train",
                               review_status="reviewed")
    manual_needs_review = FrameAnnotation(
        "manual-needs-review", .5, "manual.jpg",
        [MonsterAnnotation([10, 10, 50, 80], [30, 80], source="visual_review")],
        split="train", review_status="needs_review",
    )
    proposal = MonsterAnnotation([10, 10, 50, 80], [30, 80],
                                 source="codex_prelabel", review_required=True)
    pending = FrameAnnotation("next-work", 1, "pending.jpg", [proposal], split="train")
    later = FrameAnnotation("later", 2, "later.jpg", split="train", review_status="reviewed")
    session = ReviewSession([reviewed, manual_needs_review, pending, later], split="train",
                            queue="proposal_review_required")

    assert [item.frame_id for item in session.items] == [
        "reviewed", "later", "manual-needs-review", "next-work",
    ]
    assert session.current.frame_id == "manual-needs-review"
    assert session.navigate(-1) is True
    assert session.current.frame_id == "later"


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
