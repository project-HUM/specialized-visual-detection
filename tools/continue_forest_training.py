"""Continue a completed Forest snapshot from its latest optimizer checkpoint.

Preserves the original cosine horizon, closed mosaic, and historical patience.
The original experiment is never modified. Run this with the CUDA environment.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.forest_training_snapshot_runner import dataset_yaml, event, sha, verify


def continuation_lr(epoch, horizon=500, final_fraction=.01):
    return final_fraction + (1-final_fraction)*(1+math.cos(math.pi*min(max(epoch, 0), horizon)/horizon))/2


def historical_best(rows):
    best = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
    return int(best['epoch']), float(best['metrics/mAP50-95(B)'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.output.resolve()
    if root.exists() or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError('Output must be a new, separate snapshot')
    verify(source)
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(6)
    run_name = 'pretrained-gpu-640'
    old_run = source/'runs'/run_name
    checkpoint = max((old_run/'weights').glob('epoch*.pt'), key=lambda p: int(p.stem[5:]))
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['optimizer'] is not None and ckpt['ema'] is not None
    completed = ckpt['epoch']+1
    rows = list(csv.DictReader((old_run/'results.csv').open()))[:completed]
    assert int(rows[-1]['epoch']) == completed
    best_epoch, best_score = historical_best(rows)
    assert abs(ckpt['best_fitness']-best_score) < 1e-8
    limit, patience, horizon = 100000, 300, ckpt['train_args']['epochs']
    assert horizon == 500 and completed >= horizon-10
    for name in json.loads((source/'payload-hashes.json').read_text()):
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    shutil.copy2(checkpoint, root/'models/resume.pt')
    shutil.copy2(old_run/'weights/best.pt', root/'models/previous-best.pt')
    shutil.copy2(__file__, root/'provenance/continue_forest_training.py')
    config = json.loads((root/'training-configs.json').read_text())
    config['pretrained'].update(epochs=limit, patience=patience, resume=True)
    (root/'training-configs.json').write_text(json.dumps(config, indent=2)+'\n')
    manifest = json.loads((root/'manifest.json').read_text())
    manifest.update(experiment=root.name, continuation_parent=source.name,
        continuation=dict(checkpoint=checkpoint.name, checkpoint_sha256=sha(checkpoint),
            completed_epoch=completed, replayed_epochs=list(range(completed+1,horizon+1)),
            historical_best_epoch=best_epoch, historical_best_score=best_score,
            patience=patience, cosine_horizon=horizon, lr_after_horizon=.00001,
            ceiling=limit, rng_note='Checkpoint does not retain RNG state; replay is not bit-identical.',
            stop_metric='Overall validation mAP50-95; strictly greater score resets patience'))
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    payload = {p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(root.rglob('*')) if p.is_file()}
    (root/'payload-hashes.json').write_text(json.dumps(payload, indent=2)+'\n')
    verify(root)
    run = root/'runs'/run_name
    (run/'weights').mkdir(parents=True)
    shutil.copy2(old_run/'weights/best.pt', run/'weights/best.pt')
    with (run/'results.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)

    class ContinueTrainer(DetectionTrainer):
        def check_resume(self, overrides):
            super().check_resume(overrides)
            self.args.epochs = limit
            self.args.project = str(root/'runs')
            self.args.name = run_name
            self.args.save_dir = str(run)
            self.args.exist_ok = True
            self.args.close_mosaic = limit-(horizon-10)

        def _setup_scheduler(self):
            self.lf = lambda epoch: continuation_lr(epoch, horizon, self.args.lrf)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lf)

        def resume_training(self, checkpoint_data):
            super().resume_training(checkpoint_data)
            self.stopper.best_epoch = best_epoch
            self.stopper.best_fitness = best_score
            event(root, 'patience_restored', best_epoch=best_epoch, best_score=best_score,
                  completed_epoch=completed, no_improvement_epochs=completed-best_epoch)

        def final_eval(self):
            # Keep a full terminal checkpoint before Ultralytics strips last.pt.
            if self.last.exists():
                shutil.copy2(self.last, self.wdir/'terminal-resumable.pt')
            super().final_eval()

    event(root, 'training_started', run=run_name, mode='pretrained', resume=True,
          source_weights=str(checkpoint), source_sha256=sha(checkpoint), args=config['pretrained'])
    trainer=ContinueTrainer(overrides=dict(resume=str(root/'models/resume.pt'),
        data=str(dataset_yaml(root)), device=0, patience=patience, save_dir=str(run)))
    trainer.train()
    assert trainer.epoch+1 < limit, 'Ceiling reached before patience: continuation required'
    assert trainer.epoch+1-trainer.stopper.best_epoch >= patience
    event(root, 'training_completed', run=run_name, mode='pretrained',
          final_epoch=trainer.epoch+1, best_epoch=trainer.stopper.best_epoch,
          best_score=trainer.stopper.best_fitness, stopped_by='300 epochs without strict improvement',
          best_weights_sha256=sha(run/'weights/best.pt'))


if __name__ == '__main__':
    main()
