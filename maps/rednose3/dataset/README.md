# Specialized monster dataset

`annotations.jsonl` is the canonical source. It currently contains the 360
deterministically sampled VFR frames from the existing benchmark, all marked
`pending` with zero instances because no visual annotation has been silently
invented. Reviewers should add `MonsterAnnotation` entries for every visible or
reasonably inferable monster, including separate entries for overlapping
instances, and set `review_status` to `reviewed` only after visual audit.

The compatibility benchmark remains `../../benchmark/annotations.csv` (216 train,
72 validation, 72 sealed test). Use only train and validation during detector
development; never review or tune on the sealed test. For ambiguous overlap,
`monster-temporal-context --split train` renders previous/current/next strips.
Strict export fails if any selected annotation is pending, malformed, or marked
`review_required`. The JSONL fields preserve ground position and uncertainty
that YOLO cannot represent.

Generate pending TRAIN proposals with:

```powershell
python perception_cli.py --map rednose3 monster-prelabel --split train
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

`python perception_cli.py --map rednose3 monster-review --split train --queue pending` opens
the interactive review editor. It has VFR-aligned previous/current/next
context, a large zoomable/pannable current frame, overlapping-box selection,
drag add/move, four resize handles, Shift+click ground editing, delete, metadata
keys, per-frame undo/redo, progress totals, and autosave with one `.bak` file.
Press Enter/R to confirm and advance; confirmation alone changes the frame to
`reviewed`, changes untouched proposal provenance to
`human_confirmed_prelabel`, and clears proposal review flags. Use `E` for
`needs_review`. Navigate only with the Left/Right arrow keys. `A` toggles a
visibly indicated Add Box mode in which every left-button drag creates a new
box even over an existing box; press `A` again or `Esc` to return to Default
mode. In Default mode, `Esc` saves and quits, as does `Q` from either mode.
The `proposal_review_required` queue places reviewed frames before the
outstanding queue and opens at that boundary. All non-reviewed frames remain in
the queue, including `needs_review` frames and frames containing only manual
boxes. The Left arrow revisits and revises completed frames, while the Right
arrow continues through outstanding work without making the next launch start
over.

Two fixed-size mob box presets are configured in `review_settings.json`.
Edit each preset's visible name, source-pixel `width`/`height`, and hue in
degrees, then reopen the GUI. Drag either colored preset card from the right
panel and drop it at the monster center, or press `Ctrl+1` / `Ctrl+2` and drag
in the image; an in-image preset drag uses the drag direction while preserving
the configured size. Plain `1` through `5` still set visibility. Preset-created
annotations retain `box_preset` so their distinct hues survive save/reload.
Canonical boxes may extend beyond the source frame when at least 1/16 of their
area remains visible. The intended full box is preserved in JSONL and clipped
to visible image bounds only when YOLO labels are exported. A preset placement
preview is red while it is outside that valid region. Existing boxes preview
their destination while being moved. A preset box that starts partially out of
frame may be moved inward, but once it is fully inside, that drag and future
drags keep it contained by the frame borders.

Queue choices are `all`, `pending`, `needs_review`, and
`proposal_review_required`. Validation review remains manual and unanchored by
automatic proposals. The sealed test is rejected by proposal and review tools.

Overlapping boxes are canonical independent monster instances. The editor lets
them coexist; use Add Box mode when a drag begins over another instance. YOLO
exports one label row per box, including overlapping rows. Inference uses
permissive NMS so plausible overlaps are not discarded prematurely.
