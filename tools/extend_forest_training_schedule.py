"""Fork a verified frozen Forest dataset with a longer fresh pretrained schedule.

Never overwrite an experiment or alter its images, labels, split or initializer.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.forest_training_snapshot_runner import sha, verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=500)
    parser.add_argument("--patience",type=int,default=300)
    args=parser.parse_args()
    source,output=args.source.resolve(),args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output must be separate experiment directories")
    if args.epochs < 300 or args.patience < 300:
        raise ValueError("This extended experiment requires at least 300 epochs and patience")
    verify(source)
    configs=json.loads((source/"training-configs.json").read_text())
    if set(configs)!={"pretrained"}:
        raise ValueError("Schedule fork requires a pretrained-only source snapshot")
    payload=json.loads((source/"payload-hashes.json").read_text())
    for name in payload:
        target=output/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,target)
    configs["pretrained"].update(epochs=args.epochs,patience=args.patience)
    (output/"training-configs.json").write_text(json.dumps(configs,indent=2)+"\n")
    manifest=json.loads((output/"manifest.json").read_text())
    manifest.update(experiment=output.name,parent_experiment="combined-v3",
        schedule_comparison_experiment=source.name,
        schedule_change=f"{args.epochs} maximum epochs, patience {args.patience}; fresh standard pretrained initialization with identical seed/data/splits")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for name,destination in [("manifest.json","short-run-manifest.json"),
                             ("training-configs.json","short-run-training-configs.json")]:
        shutil.copy2(source/name,output/"provenance"/destination)
    shutil.copy2(source/"runs/pretrained-gpu-640/weights/best.pt",output/"models/short-run-best.pt")
    shutil.copy2(__file__,output/"provenance/extend_schedule.py")
    with (output/"RECOVERY.md").open("a") as stream:
        stream.write(f"\nLong schedule: {args.epochs} maximum epochs, patience {args.patience}; starts fresh\nfrom original pretrained weights. Dataset and seed match {source.name}.\n")
    files={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(output.rglob("*")) if p.is_file()}
    (output/"payload-hashes.json").write_text(json.dumps(files,indent=2)+"\n")
    backup=Path.home()/"Documents/Project HUM/Training Backups/forest-of-dead-trees-2"
    backup.mkdir(parents=True,exist_ok=True)
    archive=backup/f"{output.name}-inputs.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in output.rglob("*"):
            if p.is_file():z.write(p,p.relative_to(output).as_posix())
    archive.with_suffix(".zip.sha256").write_text(sha(archive)+"  "+archive.name+"\n")
    verify(output)
    print(output)


if __name__=="__main__":main()
