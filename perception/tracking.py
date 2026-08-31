"""Camera-compensated persistent monster tracking with supported occlusion.

Detector NMS is expected to be permissive and only limit pathological proposal
explosion. ``deduplicate_detections`` is the authoritative duplicate decision:
it requires high overlap, nearly identical centers, and similar dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, log
from typing import Any, Sequence

Box = tuple[float, float, float, float]


def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _translate(box: Box, dx: float, dy: float) -> Box:
    return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy


def _intersection(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ba = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-6, aa + ba - inter) if inter else 0.0


def _size_similarity(a: Box, b: Box) -> float:
    aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
    bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
    return min(aw, bw) / max(aw, bw) * min(ah, bh) / max(ah, bh)


def deduplicate_detections(
    detections: Sequence[dict[str, Any]],
    *,
    iou_threshold: float = 0.88,
    center_fraction: float = 0.22,
    size_similarity: float = 0.72,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Suppress only near-identical boxes; ordinary strong overlap survives."""
    retained: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    ranked = sorted(
        enumerate(detections),
        key=lambda pair: float(pair[1].get("score", 0.0)),
        reverse=True,
    )
    for fallback_index, item in ranked:
        values = item.get("box")
        if not isinstance(values, Sequence) or len(values) != 4:
            continue
        box = tuple(float(value) for value in values)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        duplicate_of: int | None = None
        for kept in retained:
            other = tuple(kept["box"])
            cx, cy = _center(box)
            ox, oy = _center(other)
            diagonal = max(1.0, hypot(other[2] - other[0], other[3] - other[1]))
            if (
                _iou(box, other) >= iou_threshold
                and hypot(cx - ox, cy - oy) <= center_fraction * diagonal
                and _size_similarity(box, other) >= size_similarity
            ):
                duplicate_of = int(kept["source_index"])
                break
        copy = dict(item)
        copy["source_index"] = int(item.get("source_index", fallback_index))
        if duplicate_of is None:
            copy["preprocessing_status"] = "retained"
            retained.append(copy)
        else:
            copy["preprocessing_status"] = "suppressed"
            copy["suppression_reason"] = "near_identical_duplicate"
            copy["suppressed_by_source_index"] = duplicate_of
            suppressed.append(copy)
    return retained, suppressed


@dataclass
class _Track:
    track_id: int
    map_box: Box
    score: float
    first_timestamp: float
    last_timestamp: float
    last_fresh_visual_timestamp: float
    last_support_timestamp: float
    min_hits: int
    hits: int = 1
    vx: float = 0.0
    vy: float = 0.0
    observed: bool = True
    supported: bool = True
    state: str = "tentative"
    occlusion_group_id: int | None = None
    last_visual_provenance: str = "visual_unknown"

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.min_hits


@dataclass
class OcclusionGroup:
    group_id: int
    member_track_ids: tuple[int, ...]
    first_timestamp: float
    last_support_timestamp: float
    supporting_detection: dict[str, Any]


class MonsterTracker:
    def __init__(
        self,
        *,
        min_hits: int = 2,
        max_gap_s: float = 0.55,
        supported_occlusion_s: float = 2.5,
        base_match_distance: float = 28.0,
        max_speed_px_s: float = 180.0,
        overlap_distance: float = 58.0,
        dedup_iou: float = 0.88,
        dedup_center_fraction: float = 0.22,
        dedup_size_similarity: float = 0.72,
    ):
        if min_hits < 1 or max_gap_s <= 0 or supported_occlusion_s <= 0:
            raise ValueError("tracker lifetimes and min_hits must be positive")
        self.min_hits = min_hits
        self.max_gap_s = max_gap_s
        self.supported_occlusion_s = supported_occlusion_s
        self.base_match_distance = base_match_distance
        self.max_speed_px_s = max_speed_px_s
        self.overlap_distance = overlap_distance
        self.dedup_iou = dedup_iou
        self.dedup_center_fraction = dedup_center_fraction
        self.dedup_size_similarity = dedup_size_similarity
        self._next_id = 1
        self._next_group_id = 1
        self._last_update_timestamp: float | None = None
        self._tracks: list[_Track] = []
        self._groups: dict[int, OcclusionGroup] = {}
        self.last_debug = self._empty_debug()

    @staticmethod
    def _empty_debug() -> dict[str, Any]:
        return {
            "input_detections": [],
            "raw_detections": [],
            "excluded_detections": [],
            "suppressed_duplicates": [],
            "retained_detections": [],
            "associations": [],
            "occlusion_groups": [],
            "tracks": [],
            "expired_track_ids": [],
        }

    def reset(self) -> None:
        self._next_id = 1
        self._next_group_id = 1
        self._last_update_timestamp = None
        self._tracks.clear()
        self._groups.clear()
        self.last_debug = self._empty_debug()

    def _predicted(self, track: _Track, now: float) -> Box:
        dt = max(0.0, now - track.last_timestamp)
        return _translate(track.map_box, track.vx * dt, track.vy * dt)

    def _compatibility(self, track: _Track, box: Box, now: float) -> tuple[bool, float]:
        predicted = self._predicted(track, now)
        px, py = _center(predicted)
        bx, by = _center(box)
        dt = max(0.0, now - track.last_timestamp)
        gate = self.base_match_distance + self.max_speed_px_s * dt
        distance = hypot(px - bx, py - by)
        overlap = _iou(predicted, box)
        return distance <= gate or overlap >= 0.08, distance / max(1.0, gate) + (1.0 - overlap) * 0.35

    def _track_lifetime(self, track: _Track) -> float:
        if track.state == "occluded" and track.supported:
            return self.supported_occlusion_s
        return self.max_gap_s

    def _prune_stale(self, now: float) -> list[int]:
        """Expire invalid identities before their motion gate can grow."""
        expired = [
            track.track_id
            for track in self._tracks
            if now - track.last_support_timestamp > self._track_lifetime(track)
        ]
        if not expired:
            return []
        expired_ids = set(expired)
        self._tracks = [track for track in self._tracks if track.track_id not in expired_ids]
        live_ids = {track.track_id for track in self._tracks}
        self._groups = {
            group_id: group
            for group_id, group in self._groups.items()
            if set(group.member_track_ids) <= live_ids
        }
        return sorted(expired)

    def _observe(self, track: _Track, item: dict[str, Any], now: float) -> None:
        box = tuple(item["box"])
        dt = max(1e-6, now - track.last_fresh_visual_timestamp)
        ox, oy = _center(track.map_box)
        nx, ny = _center(box)
        track.vx = 0.6 * track.vx + 0.4 * (nx - ox) / dt
        track.vy = 0.6 * track.vy + 0.4 * (ny - oy) / dt
        track.map_box = box
        track.score = 0.65 * track.score + 0.35 * float(item.get("score", 0.0))
        track.last_timestamp = now
        track.last_fresh_visual_timestamp = now
        track.last_support_timestamp = now
        track.hits += 1
        track.observed = True
        track.supported = True
        track.state = "visible" if track.confirmed else "tentative"
        track.occlusion_group_id = None
        track.last_visual_provenance = str(
            item.get("provenance", f"visual_{item.get('backend', 'unknown')}")
        )

    def _support_group(
        self,
        group: OcclusionGroup,
        item: dict[str, Any],
        now: float,
        by_id: dict[int, _Track],
    ) -> None:
        group.last_support_timestamp = now
        group.supporting_detection = dict(item)
        for track_id in group.member_track_ids:
            track = by_id.get(track_id)
            if track is None:
                continue
            track.map_box = self._predicted(track, now)
            track.vx *= 0.85
            track.vy *= 0.85
            track.last_timestamp = now
            track.last_support_timestamp = now
            track.observed = False
            track.supported = True
            track.state = "occluded"
            track.occlusion_group_id = group.group_id

    def _member_geometry(self, track: _Track, observation: Box, now: float) -> tuple[bool, float]:
        predicted = self._predicted(track, now)
        if _intersection(predicted, observation) <= 0.0:
            return False, float("inf")
        px, py = _center(predicted)
        ox, oy = _center(observation)
        pw = max(1.0, predicted[2] - predicted[0])
        oh = max(1.0, observation[3] - observation[1])
        horizontal_margin = max(8.0, 0.35 * pw)
        vertical_margin = max(8.0, 0.25 * oh)
        if not (
            observation[0] - horizontal_margin <= px <= observation[2] + horizontal_margin
            and observation[1] - vertical_margin <= py <= observation[3] + vertical_margin
        ):
            return False, float("inf")
        diagonal = max(1.0, hypot(observation[2] - observation[0], oh))
        outside_x = max(observation[0] - px, 0.0, px - observation[2]) / max(1.0, pw)
        score = (
            (1.0 - _iou(predicted, observation)) * 0.45
            + hypot(px - ox, py - oy) / diagonal * 0.35
            + outside_x * 0.20
        )
        return True, score

    def _select_occlusion_members(
        self, tracks: Sequence[_Track], observation: Box, now: float
    ) -> tuple[_Track, ...]:
        """Choose a plausible horizontally contiguous subset for one merged box."""
        candidates: list[tuple[_Track, float, Box]] = []
        for track in tracks:
            compatible, score = self._member_geometry(track, observation, now)
            if compatible:
                candidates.append((track, score, self._predicted(track, now)))
        candidates.sort(key=lambda entry: _center(entry[2])[0])
        if len(candidates) < 2:
            return ()

        observation_width = max(1.0, observation[2] - observation[0])
        best: tuple[float, tuple[_Track, ...]] | None = None
        for start in range(len(candidates) - 1):
            for end in range(start + 2, len(candidates) + 1):
                subset = candidates[start:end]
                left = min(entry[2][0] for entry in subset)
                right = max(entry[2][2] for entry in subset)
                span = max(1.0, right - left)
                width_ratio = observation_width / span
                if not 0.45 <= width_ratio <= 1.80:
                    continue
                edge_mismatch = (
                    abs(observation[0] - left) + abs(observation[2] - right)
                ) / max(observation_width, span)
                mean_member_score = sum(entry[1] for entry in subset) / len(subset)
                score = mean_member_score + 0.55 * edge_mismatch + 0.18 * abs(log(width_ratio))
                if best is None or score < best[0]:
                    best = (score, tuple(entry[0] for entry in subset))
        return () if best is None else best[1]

    def _group_supports_observation(
        self, group: OcclusionGroup, observation: Box, now: float, by_id: dict[int, _Track]
    ) -> bool:
        members = [by_id[track_id] for track_id in group.member_track_ids if track_id in by_id]
        if len(members) != len(group.member_track_ids):
            return False
        selected = self._select_occlusion_members(members, observation, now)
        return tuple(track.track_id for track in selected) == group.member_track_ids

    @staticmethod
    def _association_record(
        item: dict[str, Any], kind: str, track_ids: Sequence[int], cost: float | None
    ) -> dict[str, Any]:
        item["association_kind"] = kind
        item["associated_track_ids"] = list(track_ids)
        item["associated_track_id"] = track_ids[0] if len(track_ids) == 1 else None
        item["association_cost"] = None if cost is None else round(float(cost), 6)
        return {
            "source_index": item.get("source_index"),
            "kind": kind,
            "track_ids": list(track_ids),
            "cost": item["association_cost"],
        }

    def update(
        self,
        detections: Sequence[dict[str, Any]],
        timestamp: float,
        *,
        camera_origin: tuple[float, float] = (0.0, 0.0),
        exclusion_boxes: Sequence[Sequence[float]] = (),
        exclusion_iou: float = 0.35,
        include_tentative: bool = False,
    ) -> list[dict[str, Any]]:
        if self._last_update_timestamp is not None and timestamp < self._last_update_timestamp:
            raise ValueError("Tracker timestamps must be monotonic")
        self._last_update_timestamp = timestamp
        expired_track_ids = self._prune_stale(timestamp)

        ox, oy = camera_origin
        exclusions = [tuple(float(value) for value in box) for box in exclusion_boxes if len(box) == 4]
        transformed: list[dict[str, Any]] = []
        input_diagnostics: list[dict[str, Any]] = []
        excluded_detections: list[dict[str, Any]] = []
        for source_index, item in enumerate(detections):
            values = item.get("box")
            if not isinstance(values, Sequence) or len(values) != 4:
                raise ValueError("Each detection must contain box=[x1,y1,x2,y2]")
            screen = tuple(float(value) for value in values)
            diagnostic = {
                "source_index": source_index,
                "raw_bbox": list(screen),
                "raw_confidence": float(item.get("score", 0.0)),
                "backend": item.get("backend"),
                "provenance": item.get("provenance"),
            }
            if screen[2] <= screen[0] or screen[3] <= screen[1]:
                diagnostic.update(status="suppressed", suppression_reason="invalid_box")
                input_diagnostics.append(diagnostic)
                excluded_detections.append(dict(diagnostic))
                continue
            if any(_iou(screen, exclusion) >= exclusion_iou for exclusion in exclusions):
                diagnostic.update(status="suppressed", suppression_reason="player_exclusion")
                input_diagnostics.append(diagnostic)
                excluded_detections.append(dict(diagnostic))
                continue
            diagnostic["status"] = "dedup_candidate"
            input_diagnostics.append(diagnostic)
            transformed.append({**item, "box": _translate(screen, ox, oy), "source_index": source_index})

        retained, suppressed = deduplicate_detections(
            transformed,
            iou_threshold=self.dedup_iou,
            center_fraction=self.dedup_center_fraction,
            size_similarity=self.dedup_size_similarity,
        )
        diagnostic_by_index = {entry["source_index"]: entry for entry in input_diagnostics}
        for item in retained:
            diagnostic_by_index[item["source_index"]]["status"] = "retained"
        for item in suppressed:
            diagnostic_by_index[item["source_index"]].update(
                status="suppressed",
                suppression_reason=item["suppression_reason"],
                suppressed_by_source_index=item["suppressed_by_source_index"],
            )

        by_id = {track.track_id: track for track in self._tracks}
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        associations: list[dict[str, Any]] = []

        # Persist an existing group while exactly one observation supports its members.
        for group in list(self._groups.values()):
            unmatched_confirmed = sum(
                track.confirmed and track.track_id not in matched_tracks for track in self._tracks
            )
            unmatched_detections = len(retained) - len(matched_detections)
            if unmatched_detections >= unmatched_confirmed:
                break
            candidates = [
                index
                for index, item in enumerate(retained)
                if index not in matched_detections
                and self._group_supports_observation(group, tuple(item["box"]), timestamp, by_id)
            ]
            if len(candidates) == 1:
                detection_index = candidates[0]
                self._support_group(group, retained[detection_index], timestamp, by_id)
                matched_detections.add(detection_index)
                matched_tracks.update(group.member_track_ids)
                associations.append(
                    self._association_record(
                        retained[detection_index], "occlusion_support", group.member_track_ids, None
                    )
                )

        # Form only the geometrically explanatory contiguous subset, not every gated track.
        for detection_index, item in enumerate(retained):
            if detection_index in matched_detections:
                continue
            unmatched_confirmed = sum(
                track.confirmed and track.track_id not in matched_tracks for track in self._tracks
            )
            unmatched_detections = len(retained) - len(matched_detections)
            if unmatched_detections >= unmatched_confirmed:
                break
            available = [
                track
                for track in self._tracks
                if track.confirmed and track.track_id not in matched_tracks
            ]
            members = self._select_occlusion_members(available, tuple(item["box"]), timestamp)
            if len(members) < 2:
                continue
            member_ids = tuple(track.track_id for track in members)
            existing = next(
                (group for group in self._groups.values() if group.member_track_ids == member_ids),
                None,
            )
            group = existing or OcclusionGroup(
                self._next_group_id, member_ids, timestamp, timestamp, dict(item)
            )
            if existing is None:
                self._groups[group.group_id] = group
                self._next_group_id += 1
            self._support_group(group, item, timestamp, by_id)
            matched_detections.add(detection_index)
            matched_tracks.update(member_ids)
            associations.append(
                self._association_record(item, "occlusion_support", member_ids, None)
            )

        # Ordinary association is deterministic by cost, then identity and detector order.
        pairs: list[tuple[float, int, int]] = []
        for detection_index, item in enumerate(retained):
            if detection_index in matched_detections:
                continue
            for track in self._tracks:
                if track.track_id in matched_tracks:
                    continue
                compatible, cost = self._compatibility(track, tuple(item["box"]), timestamp)
                if compatible:
                    pairs.append((cost, track.track_id, detection_index))
        for cost, track_id, detection_index in sorted(pairs):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            self._observe(by_id[track_id], retained[detection_index], timestamp)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            associations.append(
                self._association_record(
                    retained[detection_index], "individual_observation", [track_id], cost
                )
            )

        for track in self._tracks:
            if track.track_id in matched_tracks:
                continue
            track.map_box = self._predicted(track, timestamp)
            track.last_timestamp = timestamp
            track.observed = False
            track.supported = False
            track.occlusion_group_id = None
            track.state = "temporal_hold" if track.confirmed else "tentative"

        # Once current support disappears, the ordinary short hold applies.
        expired_after_support_loss = self._prune_stale(timestamp)
        expired_track_ids.extend(expired_after_support_loss)
        live_ids = {track.track_id for track in self._tracks}
        self._groups = {
            group_id: group
            for group_id, group in self._groups.items()
            if set(group.member_track_ids) <= live_ids
            and any(track.occlusion_group_id == group_id for track in self._tracks)
        }

        for detection_index, item in enumerate(retained):
            if detection_index in matched_detections:
                continue
            provenance = str(
                item.get("provenance", f"visual_{item.get('backend', 'unknown')}")
            )
            box = tuple(item["box"])
            track_id = self._next_id
            self._tracks.append(
                _Track(
                    track_id,
                    box,
                    float(item.get("score", 0.0)),
                    timestamp,
                    timestamp,
                    timestamp,
                    timestamp,
                    self.min_hits,
                    state="visible" if self.min_hits == 1 else "tentative",
                    last_visual_provenance=provenance,
                )
            )
            self._next_id += 1
            associations.append(self._association_record(item, "new_track", [track_id], None))

        # Keep a single per-input diagnostic record answerable without joining
        # the retained and association lists manually.
        for item in retained:
            diagnostic_by_index[item["source_index"]].update(
                association_kind=item.get("association_kind"),
                associated_track_id=item.get("associated_track_id"),
                associated_track_ids=item.get("associated_track_ids", []),
                association_cost=item.get("association_cost"),
            )

        output: list[dict[str, Any]] = []
        track_debug: list[dict[str, Any]] = []
        for track in self._tracks:
            reference = track.last_support_timestamp
            lifetime = self._track_lifetime(track)
            confidence = (
                track.score
                if track.observed
                else track.score * max(0.0, 1.0 - (timestamp - reference) / lifetime)
            )
            confidence = max(0.0, min(1.0, confidence))
            flags = (
                []
                if track.observed
                else (["monster_overlap"] if track.supported else ["monster_temporal_hold"])
            )
            group_member_count = (
                len(self._groups[track.occlusion_group_id].member_track_ids)
                if track.occlusion_group_id in self._groups
                else None
            )
            track_debug.append(
                {
                    "track_id": track.track_id,
                    "state": track.state,
                    "age_s": round(timestamp - track.first_timestamp, 6),
                    "hits": track.hits,
                    "fresh_age_s": round(timestamp - track.last_fresh_visual_timestamp, 6),
                    "support_age_s": round(timestamp - track.last_support_timestamp, 6),
                    "occlusion_group_id": track.occlusion_group_id,
                    "group_member_count": group_member_count,
                    "confidence": round(confidence, 4),
                }
            )
            if track.state == "tentative" and not include_tentative:
                continue
            screen = _translate(track.map_box, -ox, -oy)
            output.append(
                {
                    "track_id": track.track_id,
                    "state": track.state,
                    "box": [round(value, 2) for value in screen],
                    "center": [round(value, 2) for value in _center(screen)],
                    "ground_position": [
                        round((screen[0] + screen[2]) / 2, 2),
                        round(screen[3], 2),
                    ],
                    "confidence": round(confidence, 4),
                    "confirmed": track.confirmed,
                    "observed": track.observed,
                    "supported": track.supported,
                    "occlusion_group_id": track.occlusion_group_id,
                    "group_member_count": group_member_count,
                    "last_fresh_visual_timestamp": track.last_fresh_visual_timestamp,
                    "last_support_timestamp": track.last_support_timestamp,
                    "provenance": track.last_visual_provenance if track.observed else "visual_track",
                    "quality_flags": flags,
                }
            )

        self.last_debug = {
            "input_detections": input_diagnostics,
            "raw_detections": transformed,
            "excluded_detections": excluded_detections,
            "suppressed_duplicates": suppressed,
            "retained_detections": retained,
            "associations": associations,
            "occlusion_groups": [
                {
                    "group_id": group.group_id,
                    "member_track_ids": list(group.member_track_ids),
                    "first_timestamp": group.first_timestamp,
                    "last_support_timestamp": group.last_support_timestamp,
                    "supporting_detection": group.supporting_detection,
                }
                for group in self._groups.values()
            ],
            "tracks": track_debug,
            "expired_track_ids": sorted(set(expired_track_ids)),
        }
        return sorted(output, key=lambda item: item["track_id"])

    def estimated_count(self) -> int:
        """Number of live confirmed persistent monster identities."""
        return sum(track.confirmed for track in self._tracks)

    def count_bounds(self) -> tuple[int, int]:
        """Range supported by current independent visuals and live identities.

        A visible confirmed track contributes one lower-bound evidence unit. A
        currently supported occlusion group contributes one unit regardless of
        member count. Unsupported temporal holds contribute only to the upper
        bound.
        """
        confirmed = [track for track in self._tracks if track.confirmed]
        visible_units = sum(track.state == "visible" and track.observed for track in confirmed)
        supported_group_units = len(
            {
                track.occlusion_group_id
                for track in confirmed
                if track.state == "occluded"
                and track.supported
                and track.occlusion_group_id is not None
            }
        )
        return visible_units + supported_group_units, len(confirmed)
