"""Reconstruct a CircleNet-compatible dataset; this does not restore historical crops."""
import argparse, hashlib, io, json, math, shutil
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20261111)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    canonical = json.loads(Path(__file__).with_name('canonical_train_patients.json').read_text())
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(sorted(canonical)).tolist()
    split_patients = {'train': sorted(shuffled[:20]), 'val': sorted(shuffled[20:])}
    docs = {s: {'info': {'description': 'Reconstructed MoNuSeg crops, not original CircleNet release', 'seed': args.seed},
                'licenses': [{'id': 1, 'name': 'CC BY-NC-SA 4.0', 'url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/'}],
                'images': [], 'annotations': [], 'categories': [{'id':1,'name':'nuclei','supercategory':'nuclei'}]}
            for s in ['train','val','test']}
    for s in docs: (out/s).mkdir(exist_ok=True)
    manifest = {'seed':args.seed,'sources':{},'patients':split_patients,'crops':[],'excluded_training_patients':[]}
    image_id = 0; ann_id = 0
    for source_split, source in [('train',args.train),('test',args.test)]:
        manifest['sources'][source_split] = {'url':f'https://huggingface.co/datasets/RationAI/MoNuSeg/resolve/main/data/{source_split}-00000-of-00001.parquet', 'sha256':sha(source)}
        table = pq.read_table(source)
        rows = sorted(table.to_pylist(), key=lambda r:r['patient'])
        if source_split == 'train':
            assert set(canonical).issubset({r['patient'] for r in rows})
        else:
            manifest['patients']['test'] = [r['patient'] for r in rows]
            assert len(rows)==14 and not set(canonical).intersection(manifest['patients']['test'])
        for row in rows:
            patient = row['patient']
            if source_split=='train' and patient not in canonical:
                manifest['excluded_training_patients'].append(patient); continue
            split = 'test' if source_split=='test' else ('train' if patient in split_patients['train'] else 'val')
            image = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
            assert image.size == (1000,1000)
            objects=[]
            for instance in row['instances']:
                mask=np.asarray(Image.open(io.BytesIO(instance['bytes'])))>0
                ys,xs=np.nonzero(mask)
                if len(xs):
                    x0,y0,x1,y1=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
                    objects.append((x0,y0,x1,y1,mask[y0:y1,x0:x1]))
            for crop_i in range(10):
                left,top = map(int,rng.integers(0,489,size=2))
                image_id+=1
                name=f'{patient}_crop{crop_i:02d}.png'
                patch=image.crop((left,top,left+512,top+512))
                patch.save(out/split/name)
                docs[split]['images'].append({'id':image_id,'file_name':name,'width':512,'height':512,'license':1,'source_patient':patient,'crop_xy':[left,top]})
                manifest['crops'].append({'split':split,'file':name,'patient':patient,'x':left,'y':top,'sha256':sha(out/split/name)})
                for source_index,(x0,y0,x1,y1,small) in enumerate(objects):
                    a,b,c,d=max(left,x0),max(top,y0),min(left+512,x1),min(top+512,y1)
                    if a>=c or b>=d: continue
                    clipped=small[b-y0:d-y0,a-x0:c-x0]
                    ys,xs=np.nonzero(clipped)
                    if len(xs)<3: continue
                    pts=np.column_stack((xs+a-left,ys+b-top)).astype(np.float32)
                    (cx,cy),radius=cv2.minEnclosingCircle(pts)
                    if radius<=0: continue
                    binary=np.zeros((512,512),dtype=np.uint8)
                    binary[b-top:d-top,a-left:c-left]=clipped
                    rle=mask_utils.encode(np.asfortranarray(binary));rle['counts']=rle['counts'].decode('ascii')
                    ann_id+=1
                    bbox=list(map(float,mask_utils.toBbox(rle)))
                    docs[split]['annotations'].append({'id':ann_id,'image_id':image_id,'category_id':1,'bbox':bbox,
                        'area':int(len(xs)),'segmentation':rle,'iscrowd':0,'circle_center':[float(cx),float(cy)],
                        'circle_radius':float(radius),'source_instance_index':source_index,
                        'truncated': bool(a!=x0 or b!=y0 or c!=x1 or d!=y1)})
            print(split,patient,flush=True)
    for split,doc in docs.items():
        assert len(doc['images'])=={'train':200,'val':100,'test':140}[split]
        (out/f'MoNuSeg_{split}2021.json').write_text(json.dumps(doc,separators=(',',':')))
    manifest['counts']={s:{'images':len(d['images']),'annotations':len(d['annotations'])} for s,d in docs.items()}
    manifest['source_repo_commit']='034afbca75e7b542f29e2cfdf2f7fac393137a5a'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    sample=docs['train']['images'][0]
    preview=Image.open(out/'train'/sample['file_name']).convert('RGB')
    draw=ImageDraw.Draw(preview)
    for ann in docs['train']['annotations']:
        if ann['image_id']==sample['id']:
            x,y=ann['circle_center'];r=ann['circle_radius'];draw.ellipse((x-r,y-r,x+r,y+r),outline='lime',width=1)
    preview.save(out/'annotation_preview.png')
    print(json.dumps(manifest['counts']),flush=True)

if __name__=='__main__':main()
