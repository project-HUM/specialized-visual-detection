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


def test_authoritative_dedup_retains_legitimate_iou_point_eight_overlap():
    retained, suppressed = deduplicate_detections([
        _detection([0, 0, 100, 100], .90),
        _detection([11, 0, 111, 100], .89),
    ])
    assert len(retained) == 2
    assert suppressed == []


def test_authoritative_dedup_suppresses_iou_point_ninety_five_duplicate():
    retained, suppressed = deduplicate_detections([
        _detection([0, 0, 100, 100], .90),
        _detection([2, 0, 102, 100], .89),
    ])
    assert len(retained) == 1
    assert len(suppressed) == 1
    assert suppressed[0]["suppression_reason"] == "near_identical_duplicate"


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
    tracker.update([_detection([115,95,168,155])],0.2)
    tracks=tracker.update([_detection([115,95,168,155])],1.9)
    assert len(tracks)==2
    assert all(t["state"]=="occluded" and t["supported"] and not t["observed"] for t in tracks)
    assert all(t["last_fresh_visual_timestamp"]==.1 and t["last_support_timestamp"]==1.9 for t in tracks)


def test_stale_unsupported_track_cannot_resurrect_on_distant_detection():
    tracker=MonsterTracker(min_hits=2,max_gap_s=.5,base_match_distance=20,max_speed_px_s=500)
    tracker.update([_detection([0,0,30,50])],0.0)
    tracker.update([_detection([1,0,31,50])],0.1)
    tracks=tracker.update([_detection([500,0,530,50])],1.5,include_tentative=True)
    assert [track["track_id"] for track in tracks] == [2]
    assert tracks[0]["state"] == "tentative"
    assert tracker.last_debug["expired_track_ids"] == [1]


def test_count_bounds_distinguish_visible_group_support_and_holds():
    tracker=MonsterTracker(min_hits=2,base_match_distance=50)
    first_two=[_detection([100,100,130,150]),_detection([150,100,180,150])]
    tracker.update(first_two,0.0)
    tracker.update(first_two,0.1)
    assert tracker.count_bounds() == (2,2)
    merged=tracker.update([_detection([115,95,168,155])],0.2)
    assert tracker.count_bounds() == (1,2)
    assert all(track["group_member_count"] == 2 for track in merged)

    held=tracker.update([],0.3)
    assert tracker.count_bounds() == (0,2)
    assert all(track["state"] == "temporal_hold" for track in held)
    assert all(not track["observed"] and not track["supported"] for track in held)


def test_count_bounds_for_merged_pair_plus_visible_monster_are_two_to_three():
    tracker=MonsterTracker(min_hits=2,base_match_distance=50,max_speed_px_s=80)
    boxes=[[50,100,80,150],[110,100,140,150],[170,100,200,150]]
    tracker.update([_detection(box) for box in boxes],0.0)
    tracker.update([_detection(box) for box in boxes],0.1)
    tracks=tracker.update([
        _detection([70,95,142,155]),
        _detection([172,100,202,150]),
    ],0.2)
    assert tracker.count_bounds() == (2,3)
    assert [track["state"] for track in tracks].count("occluded") == 2
    assert [track["state"] for track in tracks].count("visible") == 1


def test_unrelated_false_positive_does_not_block_local_occlusion_group():
    tracker=MonsterTracker(min_hits=2,base_match_distance=45,max_speed_px_s=80)
    boxes=[[50,100,80,150],[110,100,140,150],[250,100,280,150]]
    tracker.update([_detection(box) for box in boxes],0.0)
    tracker.update([_detection(box) for box in boxes],0.1)

    tracks=tracker.update([
        _detection([70,95,142,155],.90),       # shared M1+M2 observation
        _detection([252,100,282,150],.88),     # M3 observation
        _detection([185,40,215,90],.35),       # unrelated false positive
    ],0.2,include_tentative=True)

    grouped=[track for track in tracks if track["occlusion_group_id"] is not None]
    visible=[track for track in tracks if track["state"] == "visible"]
    tentative=[track for track in tracks if track["state"] == "tentative"]
    assert {track["track_id"] for track in grouped} == {1,2}
    assert [track["track_id"] for track in visible] == [3]
    assert [track["track_id"] for track in tentative] == [4]
    assert tracker.estimated_count() == 3
    assert tracker.count_bounds() == (2,3)


def test_unrelated_false_positive_does_not_interrupt_existing_group_support():
    tracker=MonsterTracker(
        min_hits=2,max_gap_s=.5,supported_occlusion_s=2.5,
        base_match_distance=45,max_speed_px_s=80,
    )
    boxes=[[50,100,80,150],[110,100,140,150],[250,100,280,150]]
    tracker.update([_detection(box) for box in boxes],0.0)
    tracker.update([_detection(box) for box in boxes],0.1)
    first=tracker.update([
        _detection([70,95,142,155]),
        _detection([251,100,281,150]),
    ],0.2)
    group_id=first[0]["occlusion_group_id"]

    continued=tracker.update([
        _detection([72,95,144,155],.90),
        _detection([252,100,282,150],.88),
        _detection([185,40,215,90],.35),
    ],0.7,include_tentative=True)

    grouped=[track for track in continued if track["occlusion_group_id"] == group_id]
    assert {track["track_id"] for track in grouped} == {1,2}
    assert all(track["state"] == "occluded" for track in grouped)
    assert all(track["last_support_timestamp"] == .7 for track in grouped)
    assert tracker.estimated_count() == 3
    assert tracker.count_bounds() == (2,3)


def test_distinct_local_observations_win_over_extra_shared_proposal():
    tracker=MonsterTracker(min_hits=2,base_match_distance=45,max_speed_px_s=80)
    boxes=[[50,100,80,150],[110,100,140,150]]
    tracker.update([_detection(box) for box in boxes],0.0)
    tracker.update([_detection(box) for box in boxes],0.1)

    tracks=tracker.update([
        _detection([70,95,142,155],.60),
        _detection([52,100,82,150],.90),
        _detection([112,100,142,150],.88),
    ],0.2,include_tentative=True)

    visible=[track for track in tracks if track["state"] == "visible"]
    tentative=[track for track in tracks if track["state"] == "tentative"]
    assert [track["track_id"] for track in visible] == [1,2]
    assert [track["track_id"] for track in tentative] == [3]
    assert all(track["occlusion_group_id"] is None for track in tracks)
    assert tracker.count_bounds() == (2,2)


def test_single_unsupported_hold_has_zero_to_one_bound():
    tracker=MonsterTracker(min_hits=1,max_gap_s=.5)
    tracker.update([_detection([10,10,40,60])],0.0)
    tracker.update([],0.1)
    assert tracker.count_bounds() == (0,1)


def test_nearby_third_track_is_not_absorbed_into_two_member_group():
    tracker=MonsterTracker(min_hits=2,base_match_distance=80,max_speed_px_s=100)
    boxes=[[100,100,130,150],[150,100,180,150],[190,100,220,150]]
    tracker.update([_detection(box) for box in boxes],0.0)
    tracker.update([_detection(box) for box in boxes],0.1)
    tracks=tracker.update([
        _detection([118,95,164,155]),
        _detection([191,100,221,150]),
    ],0.2)
    grouped=[track for track in tracks if track["occlusion_group_id"] is not None]
    independent=[track for track in tracks if track["occlusion_group_id"] is None]
    assert {track["track_id"] for track in grouped} == {1,2}
    assert [track["track_id"] for track in independent] == [3]


def test_three_tracks_recover_order_after_two_observations_then_split():
    tracker=MonsterTracker(min_hits=2,base_match_distance=45,max_speed_px_s=100)
    boxes=[[50,100,80,150],[110,100,140,150],[170,100,200,150]]
    tracker.update([_detection(box) for box in boxes],0.0)
    tracker.update([_detection(box) for box in boxes],0.1)
    tracker.update([_detection([70,95,142,155]),_detection([172,100,202,150])],0.2)
    recovered=tracker.update([
        _detection([55,100,85,150]),
        _detection([115,100,145,150]),
        _detection([175,100,205,150]),
    ],0.3)
    assert [track["track_id"] for track in recovered] == [1,2,3]
    assert [track["center"][0] for track in recovered] == sorted(track["center"][0] for track in recovered)


def test_three_tracks_can_share_one_supported_observation_then_recover():
    tracker=MonsterTracker(min_hits=2,base_match_distance=50,max_speed_px_s=100)
    boxes=[[50,100,80,150],[110,100,140,150],[170,100,200,150]]
    tracker.update([_detection(box) for box in boxes],0.0)
    tracker.update([_detection(box) for box in boxes],0.1)
    merged=tracker.update([_detection([70,95,182,155])],0.2)
    assert tracker.count_bounds() == (1,3)
    assert {track["group_member_count"] for track in merged} == {3}
    recovered=tracker.update([_detection(box) for box in boxes],0.3)
    assert [track["track_id"] for track in recovered] == [1,2,3]


def test_slight_motion_reversal_during_overlap_does_not_fragment_ids():
    tracker=MonsterTracker(min_hits=2,base_match_distance=45,max_speed_px_s=120)
    tracker.update([_detection([90,100,120,150]),_detection([160,100,190,150])],0.0)
    tracker.update([_detection([100,100,130,150]),_detection([150,100,180,150])],0.1)
    tracker.update([_detection([117,95,163,155])],0.2)
    recovered=tracker.update([_detection([103,100,133,150]),_detection([147,100,177,150])],0.3)
    assert [track["track_id"] for track in recovered] == [1,2]
    assert tracker.estimated_count() == 2


def test_support_loss_reverts_to_short_hold_then_expires():
    tracker=MonsterTracker(min_hits=2,max_gap_s=.5,supported_occlusion_s=2.5,base_match_distance=50)
    boxes=[_detection([100,100,130,150]),_detection([150,100,180,150])]
    tracker.update(boxes,0.0)
    tracker.update(boxes,0.1)
    tracker.update([_detection([115,95,168,155])],0.2)
    held=tracker.update([],0.6)
    assert len(held) == 2 and all(track["state"] == "temporal_hold" for track in held)
    assert tracker.update([],0.8) == []


def test_debug_metadata_explains_suppression_exclusion_and_association():
    tracker=MonsterTracker(min_hits=1)
    tracker.update([
        _detection([10,10,50,70],.95),
        _detection([11,10,51,70],.80),
        _detection([100,100,150,170],.90),
    ],0.0,exclusion_boxes=[[100,100,150,170]])
    diagnostics=tracker.last_debug
    assert diagnostics["input_detections"][1]["suppression_reason"] == "near_identical_duplicate"
    assert diagnostics["input_detections"][2]["suppression_reason"] == "player_exclusion"
    assert diagnostics["associations"][0]["kind"] == "new_track"
    assert diagnostics["retained_detections"][0]["associated_track_id"] == 1
    assert diagnostics["input_detections"][0]["associated_track_id"] == 1
    assert diagnostics["input_detections"][0]["association_cost"] is None
    assert {"age_s","hits","fresh_age_s","support_age_s"} <= set(diagnostics["tracks"][0])

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
    recovered=tracker.update([
        _detection([75,100,105,150]),
        _detection([125,100,155,150]),
    ],.3,camera_origin=(30,0))
    assert [track["track_id"] for track in recovered] == [1,2]
    assert recovered[0]["center"][0] < recovered[1]["center"][0]

def test_detector_confidence_fluctuation_does_not_fragment_track():
    tracker=MonsterTracker(min_hits=2)
    tracker.update([_detection([10,10,40,60],.95)],0)
    tracks=tracker.update([_detection([11,10,41,60],.25)],.1)
    tracks=tracker.update([_detection([12,10,42,60],.85)],.2)
    assert len(tracks)==1 and tracks[0]["track_id"]==1
    assert tracks[0]["confirmed"] is True
