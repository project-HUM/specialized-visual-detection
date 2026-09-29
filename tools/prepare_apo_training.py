"""Freeze the reviewed APO collection using a complete held-out source video."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from monster_dataset.annotation_io import export_yolo
from monster_dataset.review_app import ReviewInstanceLock
from monster_dataset.schema import read_jsonl, write_jsonl
from monster_dataset.validation import write_report
from tools.apo_training_runner import sha, write_json, inventory, verify, event

MAP=ROOT/'maps/APO'
EXPERIMENT='pretrained-v1-20260929'
CLASSES={'1':'mob','2':'hero','3':'special'}


def main():
    source=MAP/'dataset/annotations.jsonl'
    output=MAP/'dataset/runs'/EXPERIMENT
    tracked=MAP/'experiments'/EXPERIMENT
    if output.exists() or tracked.exists():
        raise FileExistsError('Experiment already exists; refusing overwrite')
    with ReviewInstanceLock(source):
        original=source.read_bytes()
        items=read_jsonl(source)
        assert len(items)==len({i.frame_id for i in items})==80
        assert all(i.review_status=='reviewed' and not i.validate() and
                   all(not b.review_required and b.box_preset in CLASSES for b in i.monsters) for i in items)
        assert sum(len(i.monsters) for i in items)==700
        order=json.loads((MAP/'dataset/review_order.json').read_text())
        assert order['frame_ids']==[i.frame_id for i in items]
        pilot=json.loads((MAP/'dataset/pilot_manifest.json').read_text())
        extra=json.loads((MAP/'dataset/codex_manual_batch_v3_manifest.json').read_text())
        anchors={f['frame_id']:{**f,'source_name':'APO_sample2','source_frame':f['frame'],
                 'source_video':pilot['source_video'],'source_video_sha256':pilot['source_video_sha256']}
                 for f in pilot['frames']}
        anchors.update({f['frame_id']:f for f in extra['frames']})
        assert set(anchors)=={i.frame_id for i in items}
        for item in items:
            anchor=anchors[item.frame_id]
            assert anchor['image_path']==item.image_path and anchor['timestamp']==item.timestamp
            assert sha(ROOT/item.image_path)==anchor['image_sha256']
        snapshot=copy.deepcopy(items)
        for item in snapshot:
            if item.frame_id=='apo-sample3-0014':
                assert 'outside_APO_town' in item.conditions
                item.split='pilot'
            else:
                item.split='validation' if anchors[item.frame_id]['source_name']=='APO_sample1' else 'train'
        assert sum(i.split=='train' for i in snapshot)==64
        assert sum(i.split=='validation' for i in snapshot)==15
        assert not ({anchors[i.frame_id]['source_name'] for i in snapshot if i.split=='train'} &
                    {anchors[i.frame_id]['source_name'] for i in snapshot if i.split=='validation'})
        output.mkdir(parents=True)
        (output/'models').mkdir()
        provenance=output/'provenance'; provenance.mkdir()
        (provenance/'annotations.canonical.jsonl').write_bytes(original)
        for name in ['review_settings.json','review_order.json','pilot_manifest.json',
                     'codex_manual_batch_v3_manifest.json']:
            shutil.copy2(MAP/'dataset'/name,provenance/name)
        for name in ['map.json','capture_manifest.json']:
            shutil.copy2(MAP/name,provenance/name)
        shutil.copy2(ROOT/'tools/prepare_apo_training.py',provenance/'prepare_apo_training.py')
        shutil.copy2(ROOT/'tools/apo_training_runner.py',output/'runner.py')
        shutil.copy2(ROOT/'yolo11n.pt',output/'models/yolo11n.pt')
        forest_config=ROOT/'maps/forest-of-dead-trees-2/experiments/combined-v4-20260928/training-configs.json'
        config=json.loads(forest_config.read_text())['pretrained']
        config['seed']=20260929
        write_json(output/'training-config.json',config)
        shutil.copy2(forest_config,provenance/'forest-training-configs.json')
        exports={split:export_yolo(snapshot,output/'dataset',split=split,image_root=ROOT,preset_classes=CLASSES)
                 for split in ['train','validation']}
        assert exports['train']['instances']==555 and exports['validation']['instances']==144
        assert set(exports['train']['classes'])==set(exports['validation']['classes'])==set(CLASSES.values())
        frames=[]
        split_hashes={'train':set(),'validation':set()}
        for item in snapshot:
            anchor=anchors[item.frame_id]
            if item.split=='pilot':
                destination=output/'excluded'/Path(item.image_path).name
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(ROOT/item.image_path,destination)
                label_path=None
            else:
                destination=output/'dataset/images'/item.split/(item.frame_id+Path(item.image_path).suffix)
                label_path=output/'dataset/labels'/item.split/(item.frame_id+'.txt')
                for line in label_path.read_text().splitlines():
                    c,x,y,w,h=map(float,line.split())
                    assert int(c)==c and 0<=c<3 and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
                split_hashes[item.split].add(sha(destination))
            item.image_path=destination.relative_to(output).as_posix()
            frames.append(dict(frame_id=item.frame_id,split=item.split,source_name=anchor['source_name'],
                source_video=anchor['source_video'],source_video_sha256=anchor['source_video_sha256'],
                source_frame=anchor['source_frame'],timestamp=item.timestamp,image_path=item.image_path,
                image_sha256=sha(destination),label_path=label_path.relative_to(output).as_posix() if label_path else None,
                label_sha256=sha(label_path) if label_path else None,boxes=len(item.monsters)))
        assert not split_hashes['train'] & split_hashes['validation']
        write_jsonl(snapshot,output/'training_annotations.jsonl')
        report=write_report(snapshot,output/'dataset-report.json')
        assert report['training_ready'] and not report['malformed_frames']
        # Stored dataset YAML is relative; runner writes a relocated absolute runtime YAML.
        (output/'dataset/dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/validation\nnames:\n  0: mob\n  1: hero\n  2: special\n')
        freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        (output/'requirements-frozen.txt').write_text(freeze,encoding='utf-8')
        import torch, ultralytics
        write_json(output/'environment.json',dict(python=sys.version,executable=sys.executable,
                   torch=torch.__version__,ultralytics=ultralytics.__version__,cuda=torch.cuda.is_available(),threads=6))
        (output/'requirements-training.txt').write_text(f'torch=={torch.__version__}\nultralytics=={ultralytics.__version__}\nopencv-python==4.11.0.86\n')
        for split in exports:
            exports[split]['output']='dataset'
        manifest=dict(schema='apo.training-snapshot.v1',experiment=EXPERIMENT,
            created_utc=datetime.now(timezone.utc).isoformat(),class_names=list(CLASSES.values()),
            initialization='generic pretrained YOLO11n',initial_weights_sha256=sha(output/'models/yolo11n.pt'),
            canonical_sha256=sha(source),exports=exports,frames=frames,
            excluded=[dict(frame_id='apo-sample3-0014',reason='Outside APO: town scene at display position 66')],
            split_policy='Entire sample1 held out; sample2 and eligible sample3 train; no source overlap',
            geometry_policy='Visually confirmed scenes; no minimap calibration or geometry required',
            backup_directory=str(Path.home()/'Documents/Project HUM/Training Backups/APO'))
        write_json(output/'manifest.json',manifest)
        (output/'RECOVERY-START-HERE.md').write_text('''# APO pretrained training recovery

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
''',encoding='utf-8')
        write_json(output/'payload-hashes.json',inventory(output))
        verify(output)
        event(output,'snapshot_created',frames=79,train=64,validation=15,canonical_sha256=sha(source))
        write_report(items,MAP/'dataset/dataset_report.json')
        assert source.read_bytes()==original
        tracked.mkdir(parents=True)
        for name in ['manifest.json','training-config.json','environment.json','requirements-training.txt',
                     'requirements-frozen.txt','payload-hashes.json','dataset-report.json','RECOVERY-START-HERE.md']:
            shutil.copy2(output/name,tracked/name)
        print(json.dumps(dict(snapshot=str(output),exports=exports),indent=2))


if __name__=='__main__':
    main()
