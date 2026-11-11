"""Regenerate annotations from included original XML using recorded crop coordinates.
No historical scores are claimed. Run from this package directory.
"""
import argparse, io, json, hashlib, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image,ImageDraw
from pycocotools import mask as mask_utils

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');args=p.parse_args()
    base=Path(args.root);root=base/'monuseg'
    manifest=json.loads((root/'manifest.json').read_text())
    docs={s:json.loads((root/f'MoNuSeg_{s}2021.json').read_text()) for s in ['train','val','test']}
    ann_id=0
    for split,doc in docs.items():
        doc['annotations']=[]
        patients=sorted({x['source_patient'] for x in doc['images']})
        source=base/'source'/('test' if split=='test' else 'train')
        for patient in patients:
            polygons=[]
            for region in ET.parse(source/(patient+'.xml')).getroot().findall('Annotation/Regions/Region'):
                pts=[(float(v.attrib['X']),float(v.attrib['Y'])) for v in region.findall('Vertices/Vertex')]
                if len(pts)>=2:polygons.append(pts)
            objects=[]
            for index,pts in enumerate(polygons):
                mask=Image.new('1',(1000,1000));ImageDraw.Draw(mask).polygon(pts,outline=True,fill=True)
                bounds=mask.getbbox()
                if bounds:
                    x0,y0,x1,y1=bounds
                    objects.append((index,x0,y0,x1,y1,np.asarray(mask.crop(bounds))))
            for im in [x for x in doc['images'] if x['source_patient']==patient]:
                left,top=im['crop_xy']; image_id=im['id']
                # Rebuild the image too, using verified original source pixels.
                image=Image.open(source/(patient+'.png')).convert('RGB')
                image.crop((left,top,left+512,top+512)).save(root/split/im['file_name'])
                for source_index,x0,y0,x1,y1,small in objects:
                    a,b,c,d=max(left,x0),max(top,y0),min(left+512,x1),min(top+512,y1)
                    if a>=c or b>=d:continue
                    clipped=small[b-y0:d-y0,a-x0:c-x0]
                    ys,xs=np.nonzero(clipped)
                    if len(xs)<3:continue
                    pts=np.column_stack((xs+a-left,ys+b-top)).astype(np.float32)
                    (cx,cy),r=cv2.minEnclosingCircle(pts)
                    binary=np.zeros((512,512),np.uint8);binary[b-top:d-top,a-left:c-left]=clipped
                    rle=mask_utils.encode(np.asfortranarray(binary));rle['counts']=rle['counts'].decode('ascii')
                    ann_id+=1
                    doc['annotations'].append({'id':ann_id,'image_id':image_id,'category_id':1,
                        'bbox':list(map(float,mask_utils.toBbox(rle))),'area':int(len(xs)),
                        'segmentation':rle,'iscrowd':0,'circle_center':[float(cx),float(cy)],
                        'circle_radius':float(r),'source_instance_index':source_index,
                        'truncated':bool(a!=x0 or b!=y0 or c!=x1 or d!=y1)})
            print(split,patient,flush=True)
        doc['info']['annotation_source']='Original downloadable MoNuSeg XML archive, rasterized with Pillow'
        (root/f'MoNuSeg_{split}2021.json').write_text(json.dumps(doc,separators=(',',':')))
    manifest['counts']={s:{'images':len(d['images']),'annotations':len(d['annotations'])} for s,d in docs.items()}
    if 'initial_mirror_sources' not in manifest: manifest['initial_mirror_sources']=manifest.get('sources',{})
    manifest['sources']=json.loads((root/'original_archive_verification.json').read_text())
    manifest['annotation_source']='Original XML from Google Drive archives; mirror labels superseded'
    for c in manifest['crops']:
        c['sha256']=hashlib.sha256((root/c['split']/c['file']).read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    im=docs['train']['images'][0];preview=Image.open(root/'train'/im['file_name']).convert('RGB');draw=ImageDraw.Draw(preview)
    for ann in docs['train']['annotations']:
        if ann['image_id']==im['id']:
            x,y=ann['circle_center'];r=ann['circle_radius'];draw.ellipse((x-r,y-r,x+r,y+r),outline='lime',width=1)
    preview.save(root/'annotation_preview.png')
    print(json.dumps(manifest['counts']),flush=True)
if __name__=='__main__':main()
