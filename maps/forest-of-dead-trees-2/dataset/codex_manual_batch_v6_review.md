# Batch 6: recent HUMAN Lich evidence

Open `../review_codex_labels_6.bat`. All 29 exact 1920x1080 source screenshots
and their 241 prelabels are pending human review. The launcher includes both
splits. No model was trained or deployed.

The 2026-09-29 review screened all 101 unique saved screenshots referenced by
the retained September 29 `lich-detections.jsonl` logs. These are deliberately
selected detector incidents, not a random sample of gameplay. The manifest
preserves the full candidate inventory, selected source paths, run IDs,
timestamps, detection events, image/log hashes, model proposals and selection
notes. Repetitive adjacent examples were reduced to 29 varied frames.

## Findings and selection

- Warning sign / snowy cliff at the left edge repeatedly resembles Lich. Frame
  `0004` is entity 618 from `ec963484`, the false positive behind the extra
  post-box clearing reversal. Frames `0007` and `0028` vary camera/banner context.
- Dying Zombies, loot and chat overlays produce false Lich proposals. They
  retain Zombie labels when a sprite is visible; loot and pets receive none.
- A Zombie behind the portal can resemble Lich (`0017`, `0054`, `0076`). Frame
  `0098` provides a real Lich behind the same kind of effect.
- Positive examples include right-edge clipping, HUD-hidden bodies, damage
  numbers, spell effects, and overlap with Zombies/Hero/pets. Frame `0094`
  shows only the Lich hat; its hidden body/feet need particularly careful review.
- Duplicate/cross-class proposals, warning signs, tree trunks, dropped items,
  and the monster-card UI were removed. Missed visible sprites were added.

Prelabels use the existing full-size Zombie/Hero/Lich presets, with estimated
visibility and inferred hidden feet. They are proposals, not confirmed ground
truth. Check crowded Zombies, fading death poses, clipping and hidden feet.
There are 194 Zombie, 30 Hero and 17 Lich boxes.

## Split assignment

| Split | Frames | Lich-positive | No-Lich hard negatives |
|---|---:|---:|---:|
| Train | 23 | 13 | 10 |
| Validation | 6 | 4 | 2 |

All selected frames from run `5be330d2` are reserved for validation
(`0056`, `0058`, `0061`, `0070`, `0073`, `0076`). All other selected source
runs go to training. No selected run appears in earlier annotation/manifest
provenance, and no exact image hash duplicates an earlier dataset image.
Prior annotations and splits were preserved byte-for-byte.

Future frames from a reserved run and all derived crops/augmentations must
inherit that run's split. A future training snapshot must honor the canonical
`split` values; do not feed batch 6 through an older script that repartitions
frames individually. This is a small curated hard-case validation set, not an
unbiased estimate of live precision/recall. Similar map scenery naturally
exists across runs; source grouping prevents adjacent-frame leakage but does
not establish independence of every visual feature. No test set was created.

The local audit previews and original annotation backup are under
`runs/codex-manual-batch-6/`. The source PNGs in `images/` are unchanged exact
copies, and the durable manifest is `codex_manual_batch_v6_manifest.json`.
