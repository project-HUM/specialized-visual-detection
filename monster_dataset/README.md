# Specialized monster dataset

`annotations.jsonl` is the canonical source. It currently contains the 360
deterministically sampled VFR frames from the existing benchmark, all marked
`pending` with zero instances because no visual annotation has been silently
invented. Reviewers should add `MonsterAnnotation` entries for every visible or
reasonably inferable monster, including separate entries for overlapping
instances, and set `review_status` to `reviewed` only after visual audit.

The compatibility benchmark remains `../benchmark/annotations.csv` (216 train,
72 validation, 72 sealed test). Use only train and validation during detector
development; never review or tune on the sealed test. For ambiguous overlap,
`monster-temporal-context --split train` renders previous/current/next strips.
Strict export fails if any selected annotation is pending, malformed, or marked
`review_required`. The JSONL fields preserve ground position and uncertainty
that YOLO cannot represent.

`python perception_cli.py monster-review --split train` opens the interactive
review UI. Drag the current image to add a source-coordinate box; the canonical
position defaults to its bottom center. The UI shows VFR-aligned previous,
current, and next frames, supports occlusion/visibility/review-required flags,
and autosaves when navigating.
