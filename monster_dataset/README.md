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

Generate pending TRAIN proposals with:

```powershell
python perception_cli.py monster-prelabel --split train
```

The default backend uses the locally cached OWLv2 model with the prompts
`pirate monster`, `pirate mushroom`, and `cartoon pirate monster`, followed by
monster-size, fixed-UI, dynamic-inventory, and conservative duplicate filters.
Use `--backend template` for the lighter capture-local template fallback. This
is not a trained specialized detector or ground truth. Every proposal has
`source=codex_prelabel`, remains `pending`, and has `review_required=true`.
The command refuses validation/test, skips reviewed frames, and conservatively
skips pending frames that already contain work. `prelabel_report.json` records
proposal counts and confidence bands.

`python perception_cli.py monster-review --split train --queue pending` opens
the interactive review editor. It has VFR-aligned previous/current/next
context, a large zoomable/pannable current frame, overlapping-box selection,
drag add/move, four resize handles, Shift+click ground editing, delete, metadata
keys, per-frame undo/redo, progress totals, and autosave with one `.bak` file.
Press Enter/R to confirm and advance; confirmation alone changes the frame to
`reviewed`, changes untouched proposal provenance to
`human_confirmed_prelabel`, and clears proposal review flags. Use `E` for
`needs_review`.

Queue choices are `all`, `pending`, `needs_review`, and
`proposal_review_required`. Validation review remains manual and unanchored by
automatic proposals. The sealed test is rejected by proposal and review tools.
