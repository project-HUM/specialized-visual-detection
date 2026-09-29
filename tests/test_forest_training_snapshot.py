import pytest
from tools.evaluate_forest_lich_snapshot import match_boxes
from tools.forest_training_snapshot_runner import require_device


def test_fixed_threshold_matches_once_and_counts_misses():
    assert match_boxes([[0,0,10,10],[0,0,10,10]],[[0,0,10,10],[30,30,40,40]]) == dict(tp=1,fp=1,fn=1)
    assert match_boxes([],[[0,0,10,10]]) == dict(tp=0,fp=0,fn=1)
    assert match_boxes([[0,0,10,10]],[]) == dict(tp=0,fp=1,fn=0)


def test_explicit_cuda_cannot_fall_back_to_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="CUDA is unavailable"):
        require_device(0)
    require_device("cpu")


def test_pretrained_resume_retains_gpu_and_rejects_wrong_mode(tmp_path, monkeypatch):
    import json
    import sys
    import ultralytics
    from tools import forest_training_snapshot_runner as runner
    root=tmp_path
    (root/"payload-hashes.json").write_text("{}")
    config={"device":0,"epochs":200}
    (root/"training-configs.json").write_text(json.dumps({"pretrained":config,"scratch":config}))
    weights=root/"runs/training/weights"
    weights.mkdir(parents=True)
    (weights/"last.pt").write_bytes(b"checkpoint")
    (weights.parent/"results.csv").write_text("epoch\n1\n")
    (root/"history.jsonl").write_text(json.dumps(dict(event="training_started",run="training",mode="pretrained",args=config))+"\n")
    calls=[]
    class Model:
        def __init__(self,path):
            assert str(path)==str(weights/"last.pt")
        def train(self,**kwargs):calls.append(kwargs)
    monkeypatch.setattr(ultralytics,"YOLO",Model)
    monkeypatch.setattr(runner,"require_device",lambda d:None)
    args=["run.py","train","--root",str(root),"--mode","pretrained","--name","training","--resume"]
    monkeypatch.setattr(sys,"argv",args)
    runner.main()
    assert calls==[dict(resume=True,device=0)]
    args[args.index("pretrained")]="scratch"
    with pytest.raises(ValueError,match="Resume mode/device"):
        runner.main()
