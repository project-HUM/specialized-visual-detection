# Temporal Occlusion Labeling Note

## Why temporal visual review finds overlapping monsters well

The useful behavior is not exceptional single-frame detection. It is semantic tracking assembled from several weak, mutually reinforcing signals:

1. Identify recurring monster silhouettes, colors, scale, hats, weapons, feet, health bars, and labels in a clean neighboring frame.
2. Preserve their platform and left-to-right ordering as the camera and sprites move.
3. Inspect the effect-covered target frame for whatever partial cues survive.
4. Check the following frame to distinguish persistent monsters from rapidly changing black/red skill effects, damage text, and particles.
5. Retain uncertainty when the pixels do not independently resolve every identity.

This works particularly well on the pirate-ship capture because monsters reuse a small sprite vocabulary, remain aligned to a few horizontal platforms, and persist longer than attack effects. A hat, feet, weapon, nameplate, or health bar can become strong evidence when combined with temporal continuity and known ordering.

## Architectural implication

The behavior should be represented as separate layers:

```text
single-frame detector
    finds independently visible monsters

temporal tracker
    preserves established monster identities

occlusion reasoning
    treats a shared visual blob as support for multiple tracks

multi-frame verifier
    checks previous/current/next crops for overlap and partial splits
```

A future temporal verifier could accept previous/current/next crops plus candidate track positions and return independently visible instance count, temporally supported instance count, overlap state, and confidence.

## Dataset implication

Do not collapse detector supervision, tracking truth, and count truth into one label. A fully hidden monster can remain valid tracker/count truth without providing learnable single-frame pixels.

Useful future fields or derived policies are:

```text
detector_eligible
    enough sprite pixels are currently visible to train/evaluate the detector

tracker_truth
    an established identity is supported by temporal evidence

count_truth
    the monster should contribute to the authoritative count
```

Low-visibility inferred boxes should train or evaluate the tracker/temporal verifier. Clean and sufficiently visible boxes should train the single-frame detector. YOLO should not be penalized for failing to detect a genuinely invisible monster.

## Recommended teacher-student loop

1. Produce temporal-context prelabels with explicit visibility, occlusion, and confidence.
2. Human-review the densest and lowest-confidence cases.
3. Train YOLO only from detector-eligible visible boxes.
4. Use before/merge/after sequences to train or test occlusion handling.
5. Let the persistent tracker convert current detector observations into stable count estimates and bounds.

The current Codex-authored results remain pending prelabels, not ground truth. Their uncertainty flags are part of the evidence and must survive human review.
