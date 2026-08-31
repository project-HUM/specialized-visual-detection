from __future__ import annotations

import pytest

from perception.tracking import MonsterTracker, deduplicate_detections


def _detection(box, score=0.8):
    return {"box": box, "score": score}


def test_one_frame_false_positive_is_not_emitted():
    tracker = MonsterTracker(min_hits=2)
    assert tracker.update([_detection([90, 90, 130, 130])], 0.0) == []
    assert tracker.update([], 0.1) == []


def test_camera_compensation_keeps_track_identity():
    tracker = MonsterTracker(min_hits=2)
    tracker.update([_detection([90, 90, 130, 130])], 0.0, camera_origin=(10, 20))
    tracks = tracker.update(
        [_detection([80, 90, 120, 130])],
        0.25,
        camera_origin=(20, 20),
    )
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 1
    assert tracks[0]["confirmed"] is True
    assert tracks[0]["center"] == [100.0, 110.0]


def test_confirmed_track_bridges_short_miss_then_expires():
    tracker = MonsterTracker(min_hits=2, max_gap_s=0.5)
    tracker.update([_detection([10, 10, 30, 30])], 0.0)
    tracker.update([_detection([12, 10, 32, 30])], 0.1)
    held = tracker.update([], 0.3)
    assert len(held) == 1
    assert held[0]["observed"] is False
    assert held[0]["provenance"] == "visual_track"
    assert held[0]["confidence"] < 0.8
    assert tracker.update([], 0.7) == []


def test_tracker_rejects_non_monotonic_timestamps():
    tracker = MonsterTracker()
    tracker.update([_detection([0, 0, 10, 10])], 1.0)
    with pytest.raises(ValueError, match="monotonic"):
        tracker.update([], 0.5)


def test_character_overlap_is_excluded_without_suppressing_nearby_monster():
    tracker = MonsterTracker(min_hits=1)
    tracks = tracker.update(
        [
            _detection([100, 100, 150, 170], 0.9),
            _detection([155, 105, 205, 175], 0.8),
        ],
        0.0,
        exclusion_boxes=[[105, 100, 150, 175]],
    )
    assert len(tracks) == 1
    assert tracks[0]["center"] == [180.0, 140.0]


def test_obvious_duplicate_is_suppressed_but_real_overlap_is_retained():
    retained, suppressed = deduplicate_detections([
        _detection([10, 10, 50, 70], .95),
        _detection([11, 10, 51, 70], .80),
        _detection([38, 10, 78, 70], .78),
    ])
    assert len(suppressed) == 1
    assert len(retained) == 2


def test_established_tracks_survive_many_to_one_occlusion_group():
    tracker = MonsterTracker(min_hits=2, base_match_distance=40, max_speed_px_s=100)
    tracker.update([_detection([100, 100, 130, 150]), _detection([150, 100, 180, 150])], 0.0)
    tracks = tracker.update([_detection([101, 100, 131, 150]), _detection([151, 100, 181, 150])], 0.1)
    assert {item["track_id"] for item in tracks} == {1, 2}
    grouped = tracker.update([_detection([118, 95, 164, 155])], 0.2)
    assert len(grouped) == 2
    assert {item["state"] for item in grouped} == {"occluded"}
    assert len({item["occlusion_group_id"] for item in grouped}) == 1
    assert tracker.estimated_count() == 2


def test_occluded_tracks_recover_without_id_swap():
    tracker = MonsterTracker(min_hits=2, base_match_distance=45, max_speed_px_s=100)
    for timestamp, detections in ((0.0, [[100, 100, 130, 150], [150, 100, 180, 150]]),
                                  (0.1, [[101, 100, 131, 150], [151, 100, 181, 150]]),
                                  (0.2, [[118, 95, 164, 155]]),
                                  (0.3, [[105, 100, 135, 150], [160, 100, 190, 150]])):
        result = tracker.update([_detection(box) for box in detections], timestamp)
    assert [item["track_id"] for item in result] == [1, 2]
    assert result[0]["center"][0] < result[1]["center"][0]

def test_supported_occlusion_has_separate_fresh_and_support_timestamps():
    tracker=MonsterTracker(min_hits=2,max_gap_s=.5,supported_occlusion_s=2.5,base_match_distance=50)
    tracker.update([_detection([100,100,130,150]),_detection([150,100,180,150])],0.0)
    tracker.update([_detection([101,100,131,150]),_detection([151,100,181,150])],0.1)
    tracks=tracker.update([_detection([115,95,168,155])],1.9)
    assert len(tracks)==2
    assert all(t["state"]=="occluded" and t["supported"] and not t["observed"] for t in tracks)
    assert all(t["last_fresh_visual_timestamp"]==.1 and t["last_support_timestamp"]==1.9 for t in tracks)

def test_two_second_occlusion_reuses_persistent_group():
    tracker=MonsterTracker(min_hits=2,max_gap_s=.5,supported_occlusion_s=2.5,base_match_distance=50)
    tracker.update([_detection([100,100,130,150]),_detection([150,100,180,150])],0)
    tracker.update([_detection([101,100,131,150]),_detection([151,100,181,150])],.1)
    group_ids=[]
    for timestamp in (.2,.7,1.2,1.7,2.2):
        tracks=tracker.update([_detection([115,95,168,155])],timestamp)
        group_ids.append(tracks[0]["occlusion_group_id"])
    assert len(set(group_ids))==1
    assert tracker.estimated_count()==2
    assert all(t["last_fresh_visual_timestamp"]==.1 for t in tracks)
    assert all(t["last_support_timestamp"]==2.2 for t in tracks)

def test_three_monster_count_survives_repeated_merge_and_separation():
    tracker=MonsterTracker(min_hits=2,max_gap_s=.5,supported_occlusion_s=2.5,base_match_distance=40,max_speed_px_s=80)
    # Pre-roll establishes the three credible tracks before the asserted six
    # frame sequence, while one-frame proposals elsewhere remain tentative.
    tracker.update([_detection(box) for box in [[49,100,79,150],[109,100,139,150],[169,100,199,150]]],-.1)
    frames=[
        [[50,100,80,150],[110,100,140,150],[170,100,200,150]],
        [[51,100,81,150],[111,100,141,150],[171,100,201,150]],
        [[70,95,142,155],[172,100,202,150]],
        [[72,95,144,155],[173,100,203,150]],
        [[74,95,146,155],[174,100,204,150]],
        [[56,100,86,150],[116,100,146,150],[176,100,206,150]],
    ]
    counts=[]
    for index,boxes in enumerate(frames):
        tracker.update([_detection(box) for box in boxes],index*.1)
        counts.append(tracker.estimated_count())
    assert counts==[3,3,3,3,3,3]

def test_unsupported_track_expires_while_supported_group_survives():
    tracker=MonsterTracker(min_hits=2,max_gap_s=.5,supported_occlusion_s=2.5,base_match_distance=50)
    tracker.update([_detection([0,0,30,50]),_detection([50,0,80,50])],0)
    tracker.update([_detection([1,0,31,50]),_detection([51,0,81,50])],.1)
    assert tracker.update([], .7)==[]

def test_camera_movement_during_occlusion_preserves_map_tracks():
    tracker=MonsterTracker(min_hits=2,base_match_distance=45)
    tracker.update([_detection([100,100,130,150]),_detection([150,100,180,150])],0,camera_origin=(0,0))
    tracker.update([_detection([90,100,120,150]),_detection([140,100,170,150])],.1,camera_origin=(10,0))
    merged=tracker.update([_detection([95,95,148,155])],.2,camera_origin=(20,0))
    assert len(merged)==2 and tracker.estimated_count()==2
    assert all(t["state"]=="occluded" for t in merged)

def test_detector_confidence_fluctuation_does_not_fragment_track():
    tracker=MonsterTracker(min_hits=2)
    tracker.update([_detection([10,10,40,60],.95)],0)
    tracks=tracker.update([_detection([11,10,41,60],.25)],.1)
    tracks=tracker.update([_detection([12,10,42,60],.85)],.2)
    assert len(tracks)==1 and tracks[0]["track_id"]==1
    assert tracks[0]["confirmed"] is True
