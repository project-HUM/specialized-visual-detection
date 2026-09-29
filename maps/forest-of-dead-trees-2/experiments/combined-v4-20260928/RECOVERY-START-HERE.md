# Start here: recover the Forest training experiment

All images, labels, splits and three initialization options are included.
Extract the archive to any directory, then open PowerShell there.
Use the Python version in environment.json and a fresh virtual environment.

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-training.txt
python run.py verify
python run.py train --mode finetune --name repeat-finetune
```

For a fresh pretrained run, use `--mode pretrained --name fresh-pretrained`.
For random weights, use `--mode scratch --name random-initialization`.
See RECOVERY.md for data layout and evaluation commands.

Use requirements-training.txt for installation. requirements-frozen.txt is the
complete environment audit and includes unrelated editable/local packages;
its installation command in the initial RECOVERY.md is superseded here.
The training-only requirements include the installed dependency closure.
Package wheels are not bundled, so recreating the environment needs package
index access; images, labels and model initialization files are self-contained.

The initial immutable payload is unchanged. recovery-addendum-hashes.json
records this environment-installation clarification and dependency list.
