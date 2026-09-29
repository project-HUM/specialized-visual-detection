# Forest combined-v4 recovery

This folder is self-contained. The original videos, HUMAN logs and repository
are not needed to train. `manifest.json` lists each image, label and split.
`annotations.jsonl` retains full boxes before YOLO edge clipping. The `dataset`
directory contains all 66 training and 16 validation images with YOLO labels.
`provenance` preserves original labels, review state, parent metrics and code state.

Use the recorded Python version (see environment.json), preferably in a fresh
virtual environment. Install `requirements-frozen.txt` with pip. The environment
was CPU PyTorch; if pip cannot find its +cpu wheel, use the PyTorch CPU index:
`python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-frozen.txt`
Package downloads still require availability of that index; wheels are not bundled.

From this recovered folder:

```powershell
python run.py verify
python run.py train --mode finetune --name repeat-finetune
python run.py train --mode pretrained --name fresh-pretrained
python run.py train --mode scratch --name random-initialization
python run.py evaluate --name repeat-finetune
```

`pretrained` starts a new training run from bundled original YOLO11n weights.
`scratch` builds the bundled architecture with random weights. It is a separate
experiment and may need more data/tuning. No old fine-tuned weights are used.
Never reuse a run name. An interrupted fine-tune can use
`python run.py train --mode finetune --name finetune-640 --resume`.
The runner verifies payload hashes before every action and resolves data paths
relative to this folder, so recovery at another directory works.

`history.jsonl`, per-run `args.yaml`, `results.csv`, plots, periodic checkpoints,
`last.pt`, and `best.pt` record training history. `evaluation` compares parent and
candidate on identical validation images. Fixed seeds/settings improve repeatability;
bitwise-identical training across different hardware/library versions is not promised.
Validation is small and selected, and is not an independent production test.
No model is automatically installed into HUMAN.
