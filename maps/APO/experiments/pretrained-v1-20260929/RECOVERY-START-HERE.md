# APO pretrained training recovery

This bundle contains the frozen PNGs, YOLO labels, reviewed source annotations,
split provenance, pretrained weights, runner, configuration, and dependencies.
The original videos and repository are not needed for training or evaluation.

Use the Python version recorded in environment.json. In a virtual environment install
the recorded CPU PyTorch version from https://download.pytorch.org/whl/cpu,
then `python -m pip install -r requirements-training.txt`.
`requirements-frozen.txt` records the complete original environment.

From this bundle directory:
```
python runner.py verify
python runner.py train --name repeat-pretrained-640
python runner.py evaluate --name repeat-pretrained-640
```
Run names must be new. For an interrupted run, use its original name with
`train --resume`; a completed checkpoint cannot be resumed as an unfinished run.
The runner resolves dataset paths after relocation and verifies immutable input
hashes before each action. Training is CPU, six threads, 640px, batch 16,
maximum 200 epochs, patience 50, seed 20260929. Classes are mob, hero, special.

Evaluation writes metrics, fixed confidence 0.25 / IoU 0.50 counts, and overlays.
The 15-image sample1 validation source is fully disjoint from training sources.
Frame 66 is retained under excluded/ but is not used for training or validation.
Canonical reviewer annotations and presets were not changed by this experiment.

Before sealing a completed run, write evaluation/<run>/visual-review.json with
the inspected overlays and observed errors, then `python runner.py archive`.
Archive defaults to run pretrained-640; pass --name for another run. The backup
location is recorded in manifest.json; the completed archive receipt records
the actual ZIP hash and independently verified restoration.
No checkpoint is automatically deployed.
