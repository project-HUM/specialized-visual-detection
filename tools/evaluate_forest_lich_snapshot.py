"""Fixed-threshold Lich audit with full-frame overlays for a portable snapshot."""
import argparse
from collections import Counter
import json
from pathlib import Path


def iou(a, b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def match_boxes(predictions, truth, threshold=.5):
    pairs=sorted(((iou(p,g),i,j) for i,p in enumerate(predictions) for j,g in enumerate(truth)),reverse=True)
    used_p,used_g=set(),set()
    for score,i,j in pairs:
        if score >= threshold and i not in used_p and j not in used_g:
            used_p.add(i);used_g.add(j)
    return dict(tp=len(used_p),fp=len(predictions)-len(used_p),fn=len(truth)-len(used_g))


def main():
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from run import verify, require_device, sha
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--name",default="pretrained-gpu-640")
    args=parser.parse_args();root=args.root.resolve();verify(root);require_device(0)
    manifest=json.loads((root/'manifest.json').read_text())
    frames=[f for f in manifest['frames'] if f['split']=='validation']
    out=root/'evaluation'/f'{args.name}-lich-conf025'
    out.mkdir(parents=True,exist_ok=False)
    report=dict(confidence=.25,nms_iou=.7,match_iou=.5,models={})
    for name,weights in [('parent',root/'models/parent-best.pt'),('candidate',root/'runs'/args.name/'weights/best.pt')]:
        model=YOLO(str(weights));records=[];panels=[]
        for frame in frames:
            im=cv2.imread(str(root/frame['image_path']));h,w=im.shape[:2];truth=[]
            for line in (root/frame['label_path']).read_text().splitlines():
                c,x,y,bw,bh=map(float,line.split())
                if int(c)==2:truth.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
            result=model.predict(im,conf=.25,iou=.7,imgsz=640,device=0,verbose=False)[0]
            predictions=[dict(box=b.xyxy[0].tolist(),confidence=float(b.conf[0])) for b in result.boxes if int(b.cls[0])==2]
            counts=match_boxes([p['box'] for p in predictions],truth)
            record=dict(frame_id=frame['frame_id'],batch6=frame['frame_id'].startswith('codex-human-batch6-'),truth=truth,predictions=predictions,**counts)
            records.append(record)
            for box in truth:
                x1,y1,x2,y2=map(round,box);cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            for p in predictions:
                x1,y1,x2,y2=map(round,p['box']);cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,255),2)
                cv2.putText(im,f'{p["confidence"]:.2f}',(x1,max(20,y1)),0,.65,(255,0,255),2)
            cv2.imwrite(str(out/f'{name}-{frame["frame_id"]}.jpg'),im)
            panel=cv2.resize(im,(640,360));header=np.zeros((35,640,3),np.uint8)
            cv2.putText(header,f'{name} {frame["frame_id"][-30:]} TP{counts["tp"]} FP{counts["fp"]} FN{counts["fn"]}',(5,24),0,.46,(255,255,255),1)
            panels.append(cv2.vconcat([header,panel]))
        summaries={}
        for subset in ['all','batch6']:
            chosen=[r for r in records if subset=='all' or r['batch6']];c=Counter()
            for r in chosen:c.update({k:r[k] for k in ['tp','fp','fn']})
            summaries[subset]=dict(frames=len(chosen),**c,precision=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else 0.,recall=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else 0.)
        report['models'][name]=dict(weights_sha256=sha(weights),summary=summaries,frames=records)
        for page,start in enumerate(range(0,len(panels),6)):
            group=panels[start:start+6];group += [np.zeros_like(panels[0])]*(6-len(group))
            cv2.imwrite(str(out/f'{name}-contact-{page}.jpg'),cv2.vconcat([cv2.hconcat(group[:3]),cv2.hconcat(group[3:])]))
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['summary'] for k,v in report['models'].items()},indent=2))


if __name__=='__main__':main()
