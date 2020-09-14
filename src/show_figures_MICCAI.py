import cv2
import numpy as np
import os
import glob
from PIL import Image


def main():
    centerNet_raw_dir = '/home/huoy1/Projects/detection/CircleNet/exp/ctdet/kidpath_dla_batch4/debug_norotate'
    centerNet_rotate_dir = '/home/huoy1/Projects/detection/CircleNet/exp/ctdet/kidpath_dla_batch4/debug_rotate'
    CircleNet_raw_dir = '/home/huoy1/Projects/detection/CircleNet/exp/circledet/kidpath_dla_batch4/debug_norotate'
    CircleNet_rotate_dir = '/home/huoy1/Projects/detection/CircleNet/exp/circledet/kidpath_dla_batch4/debug_rotate'

    output_dir = '/home/huoy1/Projects/detection/MICCAI2020/Fig3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = glob.glob(os.path.join(centerNet_raw_dir, '*_pred.png'))
    for fi in range(len(file_list)):
        pred_fname = os.path.basename(file_list[fi])
        bname = pred_fname.replace('out_pred.png', '')

        output_fname = os.path.join(output_dir, '%s_concat.png'%(bname))
        if os.path.exists(output_fname):
            continue

        cdet_gt = os.path.join(centerNet_raw_dir, '%sout_gt.png'%(bname))
        cdet_pred = os.path.join(centerNet_raw_dir, '%sout_pred.png' % (bname))
        cdet_pred_rot = os.path.join(centerNet_rotate_dir, '%sout_pred.png' % (bname))

        circledet_gt = os.path.join(CircleNet_raw_dir, '%sout_gt.png'%(bname))
        circledet_pred = os.path.join(CircleNet_raw_dir, '%sout_pred.png' % (bname))
        circledet_pred_rot = os.path.join(CircleNet_rotate_dir, '%sout_pred.png' % (bname))

        im1 = cv2.imread(cdet_gt)
        im2 = cv2.imread(cdet_pred)
        im3 = cv2.imread(cdet_pred_rot)
        im4 = cv2.imread(circledet_gt)
        im5 = cv2.imread(circledet_pred)
        im6 = cv2.imread(circledet_pred_rot)

        imc = cv2.hconcat([im1, im2, im3, im4, im5, im6])

        cv2.imwrite(output_fname, imc)










if __name__ == '__main__':

    main()