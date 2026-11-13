"""Validate image integrity, split isolation, labels, and actual CircleNet JSON readers."""
import argparse, hashlib, importlib, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as masks

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--repo',required=True);a=p.parse_args()
    root=Path(a.dataset);manifest=json.loads((root/'manifest.json').read_text())
    sys.path.insert(0,str(Path(a.repo)/'src/lib/datasets'))
    Circle=importlib.import_module('eval_protocals.kidpath_circle').CIRCLE
    seen=set();report={}
    for split,count in [('train',200),('val',100),('test',140)]:
        path=root/f'MoNuSeg_{split}2021.json';doc=json.loads(path.read_text());imgs={i['id']:i for i in doc['images']}
        assert len(imgs)==count and len(doc['images'])==count
        patients={i['source_patient'] for i in imgs.values()};assert not patients & seen;seen |= patients
        coco=COCO(str(path));circle=Circle(str(path));assert len(coco.getImgIds())==len(circle.getImgIds())==count
        for img in imgs.values():
            file=root/split/img['file_name']
            with Image.open(file) as im: assert im.size==(512,512) and im.mode=='RGB';im.verify()
        ann_ids=set();sampled=0
        for j,ann in enumerate(doc['annotations']):
            assert ann['id'] not in ann_ids;ann_ids.add(ann['id'])
            assert ann['image_id'] in imgs and ann['category_id']==1 and ann['iscrowd']==0
            x,y,w,h=ann['bbox'];cx,cy=ann['circle_center'];r=ann['circle_radius']
            assert all(math.isfinite(v) for v in [x,y,w,h,cx,cy,r])
            assert w>0 and h>0 and x>=0 and y>=0 and x+w<=512 and y+h<=512 and r>0
            assert int(masks.area(ann['segmentation']))==ann['area']
            assert np.allclose(masks.toBbox(ann['segmentation']),ann['bbox'])
            if j%100==0:
                binary=masks.decode(ann['segmentation']);ys,xs=np.nonzero(binary)
                assert np.max((xs-cx)**2+(ys-cy)**2)<= (r+0.001)**2
                sampled+=1
        assert max(len(coco.getAnnIds(imgIds=[i])) for i in imgs)<=1000
        report[split]={'images':count,'annotations':len(ann_ids),'source_patients':len(patients),
                       'sampled_mask_circle_enclosures':sampled,'coco_reader':'pass','circlenet_circle_reader':'pass'}
    for crop in manifest['crops']:
        assert hashlib.sha256((root/crop['split']/crop['file']).read_bytes()).hexdigest()==crop['sha256']
    report['all_image_checksums']='pass';report['patient_split_overlap']=0
    report['model_inference']='not run';report['historical_score_reproduction']='not claimed'
    (root/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
