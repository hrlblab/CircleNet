import numpy as np
import os
import json
import pycocotools.coco as coco
import random
import math
import fractions
import matplotlib.pyplot as plt

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

def IOU(box1, gts):
    """compute intersection over union"""
    # box1[2] = box1[0] + box1[2]
    # box1[3] = box1[1] + box1[3]
    # gts[2] = gts[0] + gts[2]
    # gts[3] = gts[1] + gts[3]

    ixmin = np.maximum(gts[0], box1[0])
    iymin = np.maximum(gts[1], box1[1])
    ixmax = np.minimum(gts[2], box1[2])
    iymax = np.minimum(gts[3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[2] - gts[0] + 1.) *
           (gts[3] - gts[1] + 1.) - inters)
    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps

def overlap_oneline(center_d_r,center_g_r,distance):
    Bx = 0.0
    Ax = 0.0+distance
    By = 0.0
    Ay = 0.0
    Ar = center_g_r+0.0
    Br = center_d_r+0.0

    d = distance
    a = Ar * Ar
    b = Br * Br
    x = (a - b + d * d) / (2 * d)
    z = x * x
    y = np.sqrt(a - z)
    overlap = a * np.arcsin(y / Ar) + b * np.arcsin(y / Br) - y * (x + np.sqrt(z + b - a))
    return overlap




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

            # r1 = 2
            # r2 = 2
            # dd = 2
            # oo1 = overlap_oneline(r1,r2,dd)
            #
            # oo2 = solve(r1, r2, dd ** 2)

    return ious




json_file = '/home/huoy1/Projects/detection/CircleNet/data/kidpath/kidneypath_test2019.json'

output_dir = '/home/huoy1/Projects/detection/MICCAI2020/Fig4'
# data_file = os.path.join(output_dir, 'data.json')
data_file = os.path.join(output_dir, 'data2.json')

random.seed(0)

coco = coco.COCO(json_file)

if os.path.exists(data_file):
    with open(data_file) as json_file:
        data = json.load(json_file)
    bious = data['bious']
    cious = data['cious']
    distances = data['distances']

    plt.plot(distances, bious, '--', linewidth=3)
    plt.plot(distances, cious, linewidth = 2)
    plt.xlabel('displacement')
    plt.ylabel('IOU')
    plt.legend(['bounding box IOU','bounding circle IOU'])
    plt.show()
    aaa=  1

else:
    bious = []
    cious = []

    for dist in range(0,1000):
        distance = dist/10.0

        bbox_iou_all = []
        circle_iou_all = []
        for key, val in coco.anns.items():
            id = val['id']
            bbox = val['bbox']
            circle = val['circle_center'] + [val['circle_radius']]

            bbox_gt = np.zeros(4)
            bbox_gt[0] = bbox[0]
            bbox_gt[1] = bbox[1]
            bbox_gt[2] = bbox[2] + bbox[0]
            bbox_gt[3] = bbox[3] + bbox[1]

            degree = random.uniform(0,360)
            x_mov = distance * math.cos(math.radians(degree))
            y_mov = distance * math.sin(math.radians(degree))

            bbox_shift = np.zeros(4)

            bbox_shift[0] = bbox[0] + x_mov
            bbox_shift[1] = bbox[1] + y_mov
            bbox_shift[2] = bbox[2] + bbox[0] + x_mov
            bbox_shift[3] = bbox[3] + bbox[1] + y_mov
            # bbox_shift[2] = bbox_shift[2] + x_mov
            # bbox_shift[3] = bbox_shift[3] + y_mov

            circle_shift = np.zeros(3)
            circle_shift[0] = circle[0] + x_mov
            circle_shift[1] = circle[1] + y_mov
            circle_shift[2] = circle[2]

            iou_box = IOU(bbox_gt, bbox_shift)
            iou_circle = circleIOU([circle], [circle_shift])
            iou_circle = iou_circle[0][0]

            bbox_iou_all.append(iou_box)
            circle_iou_all.append(iou_circle)


        bious.append(np.array(bbox_iou_all).mean())
        cious.append(np.array(circle_iou_all).mean())

    data = {}

    data['bious'] = bious
    data['cious'] = cious
    data['distances'] = list(np.array(range(0, 1000))/10)


    with open(data_file, 'w') as outfile:
        json.dump(data, outfile)


