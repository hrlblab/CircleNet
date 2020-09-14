__author__ = 'tsungyi'
import logging
import numpy as np
import pycocotools._mask as _mask
from bbox import BBox2D, BBox2DList, BBox3D
from bbox.geometry import polygon_area, polygon_collision, polygon_intersection
import sys
import math
import fractions


# Interface for manipulating masks stored in RLE format.
#
# RLE is a simple yet efficient format for storing binary masks. RLE
# first divides a vector (or vectorized image) into a series of piecewise
# constant regions and then for each piece simply stores the length of
# that piece. For example, given M=[0 0 1 1 1 0 1] the RLE counts would
# be [2 3 1 1], or for M=[1 1 1 1 1 1 0] the counts would be [0 6 1]
# (note that the odd counts are always the numbers of zeros). Instead of
# storing the counts directly, additional compression is achieved with a
# variable bitrate representation based on a common scheme called LEB128.
#
# Compression is greatest given large piecewise constant regions.
# Specifically, the size of the RLE is proportional to the number of
# *boundaries* in M (or for an image the number of boundaries in the y
# direction). Assuming fairly simple shapes, the RLE representation is
# O(sqrt(n)) where n is number of pixels in the object. Hence space usage
# is substantially lower, especially for large simple objects (large n).
#
# Many common operations on masks can be computed directly using the RLE
# (without need for decoding). This includes computations such as area,
# union, intersection, etc. All of these operations are linear in the
# size of the RLE, in other words they are O(sqrt(n)) where n is the area
# of the object. Computing these operations on the original mask is O(n).
# Thus, using the RLE can result in substantial computational savings.
#
# The following API functions are defined:
#  encode         - Encode binary masks using RLE.
#  decode         - Decode binary masks encoded via RLE.
#  merge          - Compute union or intersection of encoded masks.
#  iou            - Compute intersection over union between masks.
#  area           - Compute area of encoded masks.
#  toBbox         - Get bounding boxes surrounding encoded masks.
#  frPyObjects    - Convert polygon, bbox, and uncompressed RLE to encoded RLE mask.
#
# Usage:
#  Rs     = encode( masks )
#  masks  = decode( Rs )
#  R      = merge( Rs, intersect=false )
#  o      = iou( dt, gt, iscrowd )
#  a      = area( Rs )
#  bbs    = toBbox( Rs )
#  Rs     = frPyObjects( [pyObjects], h, w )
#
# In the API the following formats are used:
#  Rs      - [dict] Run-length encoding of binary masks
#  R       - dict Run-length encoding of binary mask
#  masks   - [hxwxn] Binary mask(s) (must have type np.ndarray(dtype=uint8) in column-major order)
#  iscrowd - [nx1] list of np.ndarray. 1 indicates corresponding gt image has crowd region to ignore
#  bbs     - [nx4] Bounding box(es) stored as [x y w h]
#  poly    - Polygon stored as [[x1 y1 x2 y2...],[x1 y1 ...],...] (2D list)
#  dt,gt   - May be either bounding boxes or encoded masks
# Both poly and bbs are 0-indexed (bbox=[0 0 1 1] encloses first pixel).
#
# Finally, a note about the intersection over union (iou) computation.
# The standard iou of a ground truth (gt) and detected (dt) object is
#  iou(gt,dt) = area(intersect(gt,dt)) / area(union(gt,dt))
# For "crowd" regions, we use a modified criteria. If a gt object is
# marked as "iscrowd", we allow a dt to match any subregion of the gt.
# Choosing gt' in the crowd gt that best matches the dt can be done using
# gt'=intersect(dt,gt). Since by definition union(gt',dt)=dt, computing
#  iou(gt,dt,iscrowd) = iou(gt',dt) = area(intersect(gt,dt)) / area(dt)
# For crowd gt regions we use this modified criteria above for the iou.
#
# To compile run "python setup.py build_ext --inplace"
# Please do not contact us for help with compiling.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

iou         = _mask.iou
merge       = _mask.merge
frPyObjects = _mask.frPyObjects

def encode(bimask):
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]

def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]

def area(rleObjs):
    if type(rleObjs) == list:
        return _mask.area(rleObjs)
    else:
        return _mask.area([rleObjs])[0]

def toBbox(rleObjs):
    if type(rleObjs) == list:
        return _mask.toBbox(rleObjs)
    else:
        return _mask.toBbox([rleObjs])[0]


def iou_3d(a: BBox3D, b: BBox3D):
    """
    Compute the Intersection over Union (IoU) of a pair of 3D bounding boxes.

    Alias for `jaccard_index_3d`.
    """
    return jaccard_index_3d(a, b)



def jaccard_index_3d(a: BBox3D, b: BBox3D):
    """
    Compute the Jaccard Index / Intersection over Union (IoU) of a pair of 3D bounding boxes.
    We compute the IoU using the top-down bird's eye view of the boxes.

    **Note**: We follow the KITTI format and assume only yaw rotations (along z-axis).

    Args:
        a (:py:class:`BBox3D`): 3D bounding box.
        b (:py:class:`BBox3D`): 3D bounding box.

    Returns:
        :py:class:`float`: The IoU of the 2 bounding boxes.
    """
    # check if the two boxes don't overlap
    if not polygon_collision(a.p[0:4, 0:2], b.p[0:4, 0:2]):
        return np.round_(0, decimals=5)

    intersection_points = polygon_intersection(a.p[0:4, 0:2], b.p[0:4, 0:2])
    inter_area = polygon_area(intersection_points)

    zmax = np.minimum(a.cz, b.cz)
    zmin = np.maximum(a.cz - a.h, b.cz - b.h)

    inter_vol = inter_area * np.maximum(0, zmax-zmin)

    a_vol = a.l * a.w * a.h
    b_vol = b.l * b.w * b.h

    union_vol = (a_vol + b_vol - inter_vol)

    iou = inter_vol / union_vol

    # set nan and +/- inf to 0
    if np.isinf(iou) or np.isnan(iou):
        iou = 0

    return np.round_(iou, decimals=5)


def dddIOU(d,g):
    # d = ddd_to_dd_box(d)
    ious = np.zeros((len(d), len(g)))
    for di in range(len(d)):
        box_d = BBox3D(d[di][0],d[di][1],d[di][2],d[di][3],d[di][4],d[di][5])
        for gi in range(len(g)):
            box_g = BBox3D(g[gi][0],g[gi][1],g[gi][2],g[gi][3],g[gi][4],g[gi][5])
            ious[di, gi] = iou_3d(box_d,box_g)
    return ious

def circleBoxIOU(d,g):
    ious = np.zeros((len(d), len(g)))
    for di in range(len(d)):
        center_d_x = d[di][0]
        center_d_y = d[di][1]
        center_d_r = d[di][2]
        for gi in range(len(g)):
            center_g_x = g[gi][0]
            center_g_y = g[gi][1]
            center_g_r = g[gi][2]
            distance = math.sqrt((center_d_x - center_g_x)**2 + (center_d_y - center_g_y)**2)
            if center_d_r <=0 or center_g_r <=0 or distance > (center_d_r + center_g_r) :
                ious[di, gi] = 0
            else:
                box_d = np.array([0, 0, 0, 0], dtype=np.float32)
                box_d[0] = 512.0 - center_d_r
                box_d[1] = 512.0 - center_d_r
                box_d[2] = center_d_r*2
                box_d[3] = center_d_r*2

                box_g = np.array([0, 0, 0, 0], dtype=np.float32)
                box_g[0] = 512.0 - center_g_r
                box_g[1] = 512.0 + distance - center_g_r
                box_g[2] = center_g_r*2
                box_g[3] = center_g_r*2

                overlap = solve(center_d_r, center_g_r, distance ** 2)
                union = math.pi * (center_d_r**2) + math.pi * (center_g_r**2) -  overlap
                if union == 0:
                    ious[di,gi] = 0
                else:
                    iscrowd = [0, 0]
                    ious[di, gi] = iou([list(box_d)], [list(box_g)], iscrowd)
                    # print(ious[di, gi])

    return ious

def circleIOU(d,g):
    ious = np.zeros((len(d), len(g)))
    for di in range(len(d)):
        center_d_x = d[di][0]
        center_d_y = d[di][1]
        center_d_r = d[di][2]
        for gi in range(len(g)):
            center_g_x = g[gi][0]
            center_g_y = g[gi][1]
            center_g_r = g[gi][2]
            distance = math.sqrt((center_d_x - center_g_x)**2 + (center_d_y - center_g_y)**2)
            if center_d_r <=0 or center_g_r <=0 or distance > (center_d_r + center_g_r) :
                ious[di, gi] = 0
            else:
                overlap = solve(center_d_r, center_g_r, distance**2)
                union = math.pi * (center_d_r**2) + math.pi * (center_g_r**2) -  overlap
                if union == 0:
                    ious[di,gi] = 0
                else:
                    ious[di, gi] = overlap/union
    return ious


def f(x):
    """
    Compute  x - sin(x) cos(x)  without loss of significance
    """
    if abs(x) < 0.01:
        return 2 * x ** 3 / 3 - 2 * x ** 5 / 15 + 4 * x ** 7 / 315
    return x - math.sin(x) * math.cos(x)


def acos_sqrt(x, sgn):
    """
    Compute acos(sgn * sqrt(x)) with accuracy even when |x| is close to 1.
    http://www.wolframalpha.com/input/?i=acos%28sqrt%281-y%29%29
    http://www.wolframalpha.com/input/?i=acos%28sqrt%28-1%2By%29%29
    """
    assert isinstance(x, fractions.Fraction)

    y = 1 - x
    if y < 0.01:
        # pp('y < 0.01')
        numers = [1, 1, 3, 5, 35]
        denoms = [1, 6, 40, 112, 1152]
        ans = fractions.Fraction('0')
        for i, (n, d) in enumerate(zip(numers, denoms)):
            ans += y ** i * n / d
        assert isinstance(y, fractions.Fraction)
        ans *= math.sqrt(y)
        if sgn >= 0:
            return ans
        else:
            return math.pi - ans

    return math.acos(sgn * math.sqrt(x))


def solve(r1, r2, d_squared):
    r1, r2 = min(r1, r2), max(r1, r2)

    d = math.sqrt(d_squared)
    if d >= r1 + r2:  # circles are far apart
        return 0.0
    if r2 >= d + r1:  # whole circle is contained in the other
        return math.pi * r1 ** 2

    r1f, r2f, dsq = map(fractions.Fraction, [r1, r2, d_squared])
    r1sq, r2sq = map(lambda i: i * i, [r1f, r2f])
    numer1 = r1sq + dsq - r2sq
    cos_theta1_sq = numer1 * numer1 / (4 * r1sq * dsq)
    numer2 = r2sq + dsq - r1sq
    cos_theta2_sq = numer2 * numer2 / (4 * r2sq * dsq)
    theta1 = acos_sqrt(cos_theta1_sq, math.copysign(1, numer1))
    theta2 = acos_sqrt(cos_theta2_sq, math.copysign(1, numer2))
    result = r1 * r1 * f(theta1) + r2 * r2 * f(theta2)

    # pp("d = %.16e" % d)
    # pp("cos_theta1_sq = %.16e" % cos_theta1_sq)
    # pp("theta1 = %.16e" % theta1)
    # pp("theta2 = %.16e" % theta2)
    # pp("f(theta1) = %.16e" % f(theta1))
    # pp("f(theta2) = %.16e" % f(theta2))
    # pp("result = %.16e" % result)

    return result


def pp(*args, **kwargs):
    return print(*args, file=sys.stderr, **kwargs)