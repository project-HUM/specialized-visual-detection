# Specialized monster dataset

`annotations.jsonl` is the canonical source. It currently contains the 360
deterministically sampled VFR frames from the existing benchmark, all marked
`pending` with zero instances because no visual annotation has been silently
invented. Reviewers should add `MonsterAnnotation` entries for every visible or
reasonably inferable monster, including separate entries for overlapping
instances, and set `review_status` to `reviewed` only after visual audit.

The compatibility benchmark remains `../benchmark/annotations.csv` (216 train,
72 validation, 72 sealed test). Detector training should use a contiguous-time
split derived from this source once labels exist; never tune on the sealed test.
Use the contact-sheet command for review, then regenerate the report and export
YOLO labels. The JSONL fields preserve ground position and uncertainty that
YOLO cannot represent.
