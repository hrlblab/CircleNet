import numpy as np
from scipy import interpolate

def sens_at_FP(boxes_all, gts_all, avgFP, iou_th):
    """compute the sensitivity at avgFP (average FP per image)"""
    sens, fp_per_img = FROC_part_det(boxes_all, gts_all, iou_th)
    avgFP_in = [a for a in avgFP if a <= fp_per_img[-1]]
    avgFP_out = [a for a in avgFP if a > fp_per_img[-1]]
    f = interpolate.interp1d(fp_per_img, sens)
    res = np.hstack([f(np.array(avgFP_in)), np.ones((len(avgFP_out, )))*sens[-1]])
    return res

def sens_at_FP_3d(boxes_all, gts_all, avgFP, iou_th):
    """compute the sensitivity at avgFP (average FP per image)"""
    sens, fp_per_img, nMiss, nMissinds = FROC_3D(boxes_all, gts_all, iou_th)
    avgFP_in = [a for a in avgFP if a <= fp_per_img[-1]]
    avgFP_out = [a for a in avgFP if a > fp_per_img[-1]]
    f = interpolate.interp1d(fp_per_img, sens)
    res = np.hstack([f(np.array(avgFP_in)), np.ones((len(avgFP_out, )))*sens[-1]])
    return res

def miss_tumor_3d(boxes_all, gts_all, avgFP, iou_th):
    """compute the sensitivity at avgFP (average FP per image)"""
    sens, fp_per_img, nMiss, nMissinds  = FROC_3D(boxes_all, gts_all, iou_th)
    return nMiss, nMissinds

def miss_tumor_2d(boxes_all, gts_all, avgFP, iou_th):
    """compute the sensitivity at avgFP (average FP per image)"""
    sens, fp_per_img, nMiss = FROC_part_det(boxes_all, gts_all, iou_th)
    return nMiss

def FROC(boxes_all, gts_all, iou_th):
    """Compute the Free ROC curve, for single class only"""
    nImg = len(boxes_all)
    img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs[ord]
    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    for i in range(len(boxes_cat)):
        overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
        if len(overlaps) == 0 or overlaps.max() < iou_th:
            nMiss += 1
        else:
            for j in range(len(overlaps)):
                if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1
        tps.append(nHits)
        fps.append(nMiss)
    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg
    return sens, fp_per_img

def FROC_part_det(boxes_all, gts_all, iou_th):
    """Compute the Free ROC curve, for single class only.
    When a box detects a part of a GT (IOU<th, IoBB>=th), don't consider it as FP."""
    nImg = len(boxes_all)
    img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs[ord]
    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    for i in range(len(boxes_cat)):
        iou, iobb = IOU_IOBB(boxes_cat[i, :], gts_all[img_idxs[i]])
        if len(iou) == 0 or (iou.max() < iou_th and iobb.max() < iou_th):
            nMiss += 1
        else:
            for j in range(len(iou)):
                if iou[j] >= iou_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1
        tps.append(nHits)
        fps.append(nMiss)
    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg
    return sens, fp_per_img

def FROC_3D(boxes_all, gts_all, iou_th):
    """Compute the Free ROC curve of 3D boxes, for single class only"""
    nImg = len(boxes_all)
    img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :6]
    img_idxs = img_idxs[ord]
    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    nMissinds = []
    tps = []
    fps = []
    # print('Using IOBB for FROC 3D')
    for i in range(len(boxes_cat)):
        iou, iobb = IOU_IOBB_3D(boxes_cat[i, :], gts_all[img_idxs[i]])
        # overlaps = IOU_3D(boxes_cat[i, :], gts_all[img_idxs[i]])
        if len(iou) == 0 or (iou.max() < iou_th and iobb.max() < iou_th):
            nMiss += 1
            nMissinds.append(img_idxs[i])
        else:
            for j in range(len(iou)):
                if iobb[j] >= iou_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1
        tps.append(nHits)
        fps.append(nMiss)

    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg
    return sens, fp_per_img, nMiss, nMissinds

def IOU(box1, gts):
    """compute intersection over union"""
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[:, 2] - gts[:, 0] + 1.) *
           (gts[:, 3] - gts[:, 1] + 1.) - inters)
    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps

def IOU_IOBB(box1, gts):
    """compute intersection over union and bbox"""
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    box_size = (box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.)
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[:, 2] - gts[:, 0] + 1.) *
           (gts[:, 3] - gts[:, 1] + 1.) - inters)
    iobb = inters / box_size
    iou = inters / uni
    return iou, iobb

def IOU_single_side(box1, gts):
    """compute intersection over boxes and gts"""
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    box_size = (box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.)
    gt_sizes = (gts[:, 2] - gts[:, 0] + 1.) * (gts[:, 3] - gts[:, 1] + 1.)
    iobb = inters / box_size
    iogts = inters / gt_sizes
    return iobb, iogts

def IOU_3D(box1, gts):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    izmin = np.maximum(gts[:, 2], box1[2])
    ixmax = np.minimum(gts[:, 3], box1[3])
    iymax = np.minimum(gts[:, 4], box1[4])
    izmax = np.minimum(gts[:, 5], box1[5])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    id = np.maximum(izmax - izmin + 1., 0.)
    inters = iw * ih * id
    # union
    uni = ((box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.) * (box1[5] - box1[2] + 1.) +
           (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.) * (gts[:, 5] - gts[:, 2] + 1.) - inters)
    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps

def IOU_IOBB_3D(box1, gts):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    izmin = np.maximum(gts[:, 2], box1[2])
    ixmax = np.minimum(gts[:, 3], box1[3])
    iymax = np.minimum(gts[:, 4], box1[4])
    izmax = np.minimum(gts[:, 5], box1[5])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    id = np.maximum(izmax - izmin + 1., 0.)
    inters = iw * ih * id
    # union
    uni = ((box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.) * (box1[5] - box1[2] + 1.) +
           (gts[:, 3] - gts[:, 0] + 1.) * (gts[:, 4] - gts[:, 1] + 1.) * (gts[:, 5] - gts[:, 2] + 1.) - inters)
    box_size = (box1[3] - box1[0] + 1.) * (box1[4] - box1[1] + 1.) * (box1[5] - box1[2] + 1.)
    iobb = inters / box_size
    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps, iobb