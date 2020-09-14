from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import numpy as np
import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory
import openslide
import xmltodict
from PIL import Image
from utils.debugger import Debugger
from external.nms import soft_nms
from lib.datasets.eval_protocals.mask import circleIOU

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
scn_ext = ['scn','sys','svs']


def get_none_zero(black_arr):

    nonzeros = black_arr.nonzero()
    starting_y = nonzeros[0].min()
    ending_y = nonzeros[0].max()
    starting_x = nonzeros[1].min()
    ending_x = nonzeros[1].max()

    return starting_x, starting_y, ending_x, ending_y

def scan_nonblack_end(simg,px_start,py_start,px_end,py_end):
    offset_x = 0
    offset_y = 0
    line_x = py_end-py_start
    line_y = px_end-px_start

    val = simg.read_region((px_end+offset_x, py_end), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end+offset_x, py_end), 0, (1, line_x))
        arr = np.array(val)[:, :, 0].sum()
        offset_x = offset_x + 1

    val = simg.read_region((px_end, py_end+offset_y), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end, py_end+offset_y), 0, (line_y, 1))
        arr = np.array(val)[:, :, 0].sum()
        offset_y = offset_y + 1

    x = px_end+(offset_x-1)
    y = py_end+(offset_y-1)
    return x,y


def get_nonblack_ending_point(simg):
    px = 0
    py = 0
    black_img = simg.read_region((px, py), 3, (3000, 3000))
    starting_x, starting_y, ending_x, ending_y = get_none_zero(np.array(black_img)[:, :, 0])

    multiples = int(np.floor(simg.level_dimensions[0][0]/float(simg.level_dimensions[3][0])))

    #staring point
    px2 = (starting_x - 1) * multiples
    py2 = (starting_y - 1) * multiples
    #ending point
    px3 = (ending_x - 1) * (multiples-1)
    py3 = (ending_y - 1) * (multiples-1)

    # black_img_big = simg.read_region((px2, py2), 0, (1000, 1000))
    # offset_x, offset_y, offset_xx, offset_yy = get_none_zero(np.array(black_img_big)[:, :, 0])
    #
    # x = px2+offset_x
    # y = py2+offset_y

    xx, yy = scan_nonblack_end(simg, px2, py2, px3, py3)

    return xx,yy

def scn_to_patchs(scn_file, working_dir, opt):

    patch_2d_dir = os.path.join(working_dir, '2d_patch')
    if not os.path.exists(patch_2d_dir):
        os.makedirs(patch_2d_dir)

    simg = openslide.open_slide(scn_file)

    # start_xx, start_yy = get_nonblack_starting_point(simg)
    try:
        start_x = np.int(simg.properties['openslide.bounds-x'])
        start_y = np.int(simg.properties['openslide.bounds-y'])
        width_x = np.int(simg.properties['openslide.bounds-width'])
        height_y = np.int(simg.properties['openslide.bounds-height'])
    except:
        start_x = 0
        start_y = 0
        width_x = np.int(simg.properties['aperio.OriginalWidth'])
        height_y = np.int(simg.properties['aperio.OriginalHeight'])
    end_x = start_x + width_x
    end_y = start_y + height_y

    input_width = opt.input_w
    input_height = opt.input_h

    down_rate = simg.level_downsamples[opt.lv]
    resolution = simg.level_dimensions[opt.lv]

    num_patch_x_lv = np.int(np.ceil(width_x/down_rate/input_width))
    num_patch_y_lv = np.int(np.ceil(height_y/down_rate/input_height))

    if opt.lv >= 2:
        img_lv2_width = num_patch_x_lv * input_width
        img_lv2_height = num_patch_y_lv * input_height

        img_lv2 = simg.read_region((start_x, start_y), 2, (img_lv2_width, img_lv2_height))
        img = np.array(img_lv2.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # opencv大坑之BGR opencv对于读进来的图片的通道排列是BGR，而不是主流的RGB！谨记！
    else:
        patch_size = 32
        num_patch_x_lv_patch = np.int(np.ceil(width_x / down_rate / patch_size))
        num_patch_y_lv_patch  = np.int(np.ceil(height_y / down_rate / patch_size))
        whole_width_lv_patch  = num_patch_x_lv_patch  * patch_size
        whole_height_lv_patch  = num_patch_y_lv_patch  * patch_size
        img = np.zeros((whole_height_lv_patch , whole_width_lv_patch , 3), dtype=np.uint8)

        for xi in range(num_patch_y_lv_patch ):
            for yi in range(num_patch_x_lv_patch ):
                low_res_offset_x = np.int(xi * patch_size)
                low_res_offset_y = np.int(yi * patch_size)

                patch_start_x = start_x + np.int(low_res_offset_y * down_rate)
                patch_start_y = start_y + np.int(low_res_offset_x * down_rate)
                img_lv = simg.read_region((patch_start_x, patch_start_y), opt.lv, (patch_size, patch_size))
                img_lv = np.array(img_lv.convert('RGB'))
                if (low_res_offset_x + patch_size) <= whole_height_lv_patch  and (
                        low_res_offset_y + patch_size) <= whole_width_lv_patch :
                    img[low_res_offset_x:(low_res_offset_x + patch_size),
                    low_res_offset_y:(low_res_offset_y + patch_size), :] = img_lv
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    num_overlap_patch_x = 2 * num_patch_x_lv - 1
    num_overlap_patch_y = 2 * num_patch_y_lv - 1

    for xi in range(num_overlap_patch_x):
        for yi in range(num_overlap_patch_y):

            if xi == 2 and yi == 13:
                aaa = 1

            low_res_offset_x = np.int(xi * input_width / 2)
            low_res_offset_y = np.int(yi * input_height / 2)

            patch_start_x = start_x + np.int(low_res_offset_x * down_rate)
            patch_start_y = start_y + np.int(low_res_offset_y * down_rate)
            img_lv = simg.read_region((patch_start_x, patch_start_y), opt.lv, (input_width, input_height))
            img_patch = np.array(img_lv.convert('RGB'))
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)


            # img_patch = img[patch_start_y:(patch_start_y+input_height),patch_start_x:(patch_start_x+input_width),:]
            patch_file_bname = 'lv%d-x-%04d-x-%04d-x-%d-x-%d.png' % (opt.lv,xi,yi,low_res_offset_x,low_res_offset_y)
            patch_file = os.path.join(patch_2d_dir, patch_file_bname)
            cv2.imwrite(patch_file, img_patch)

            # img_out = Image.fromarray(img_patch)
            # img_out.save(patch_file)

    return patch_2d_dir, img, simg



def nms_circle(input, iou_th, merge_overlap=False):
    boxes = np.zeros((len(input), 5))
    scores = np.zeros(len(input))
    for di in range(len(input)):
        bbox_d = input[di]
        boxes[di, 0] = bbox_d[0]
        boxes[di, 1] = bbox_d[1]
        boxes[di, 2] = bbox_d[2]
        boxes[di, 3] = bbox_d[3]
        boxes[di, 4] = bbox_d[4]
        scores[di] = bbox_d[3]

    if len(scores) == 0:
       return np.hstack((boxes, scores))
    ord = np.argsort(scores)[::-1]
    scores = scores[ord]
    boxes = boxes[ord]
    sel_boxes = boxes[0][None]
    sel_scores = scores[0:1]
    for i in range(1, len(scores)):
       ious = circleIOU([boxes[i]], sel_boxes)
       if ious.max() < iou_th:
           sel_boxes = np.vstack((sel_boxes, boxes[i]))
           sel_scores = np.hstack((sel_scores, scores[i]))
       else:
           if merge_overlap:
               idx = ious.argmax()
               dim = sel_boxes.shape[1]//2
               sel_boxes[idx, :dim] = np.minimum(sel_boxes[idx, :dim], boxes[i][:dim])
               sel_boxes[idx, dim:] = np.maximum(sel_boxes[idx, dim:], boxes[i][dim:])

    # np.hstack((sel_boxes, sel_scores[:, None]))

    output = np.zeros((len(sel_boxes), 7))
    for di in range(len(output)):
        bbox_d = sel_boxes[di]
        output[di, 0] = bbox_d[0]
        output[di, 1] = bbox_d[1]
        output[di, 2] = bbox_d[2]
        output[di, 3] = bbox_d[3]
        output[di, 4] = bbox_d[4]
    return output


def merge_outputs(num_classes, max_per_image, run_nms, detections):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.array(detections, dtype=np.float32)

        if run_nms:
            results[j] = nms_circle(results[j], iou_th=0.35)
    scores = np.hstack(
        [results[j][:, 3] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 3] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def run_one_scn(demo_scn,demo_dir,opt):
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    basename = os.path.basename(demo_scn)
    basename = basename.replace('.scn', '')
    basename = basename.replace('.svs', '')
    # basename = basename.replace(' ', '-')
    working_dir = os.path.join(demo_dir, basename)

    xml_file = os.path.join(working_dir, '%s.xml' % (basename))
    if os.path.exists(xml_file):
        return

    patch_2d_dir, simg_big, simg = scn_to_patchs(demo_scn, working_dir, opt)

    if os.path.isdir(patch_2d_dir):
        image_names = []
        ls = os.listdir(patch_2d_dir)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(patch_2d_dir, file_name))
    else:
        image_names = [patch_2d_dir]

    detect_all = None
    count = 1
    for (image_name) in image_names:
        ret = detector.run(image_name)
        results = ret['results']
        res_strs = os.path.basename(image_name).replace('.png', '').split('-x-')
        lv_str = res_strs[0]
        patch_start_x = np.int(res_strs[3])
        patch_start_y = np.int(res_strs[4])

        if opt.filter_boarder:
            output_h = opt.input_h  # hard coded
            output_w = opt.input_w  # hard coded
            for j in range(1, opt.num_classes + 1):
                for i in range(len(results[j])):
                    cp = [0, 0]
                    cp[0] = results[j][i][0]
                    cp[1] = results[j][i][1]
                    cr = results[j][i][2]
                    if cp[0] - cr < 0 or cp[0] + cr > output_w:
                        results[j][i][3] = 0
                        continue
                    if cp[1] - cr < 0 or cp[1] + cr > output_h:
                        results[j][i][3] = 0
                        continue

        for j in range(1, opt.num_classes + 1):
            for circle in results[j]:
                if circle[3] > opt.vis_thresh:
                    circle_out = circle[:]
                    circle_out[0] = circle[0] + patch_start_x
                    circle_out[1] = circle[1] + patch_start_y
                    if detect_all is None:
                        detect_all = [circle]
                    else:
                        detect_all = np.append(detect_all, [circle], axis=0)

        time_str = ''
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(' %d/%d %s'%(count, len(image_names),time_str))
        count = count+1

    num_classes = 1
    scales = 1
    max_per_image = 2000
    run_nms = True
    results2 = merge_outputs(num_classes, max_per_image, run_nms, detect_all)
    detect_all = results2[1]

    # detections = []
    # det_clss = {}
    # det_clss[1] = detect_all
    # detections.append(det_clss)
    # detect_all = merge_outputs(opt, detections)

    if not simg_big is None:
        debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug == 3),
                            theme=opt.debugger_theme)
        debugger.add_img(simg_big, img_id='')
        debugger.save_all_imgs(working_dir, prefix='%s' % (basename))  # save original image
        json_file = os.path.join(working_dir,'%s.json' % (basename))
        debugger.save_detect_all_to_json(simg_big, detect_all, json_file, opt, simg)

        for circle in detect_all:
            debugger.add_coco_circle(circle[:3], circle[-1],
                                     circle[3], img_id='')
        debugger.save_all_imgs(working_dir, prefix='%s_overlay' % (basename))  # save original overlay

    # # make open slide file
    # with open("/media/huoy1/48EAE4F7EAE4E264/Projects/detection/test_demo/Case 01-3_manual_good.xml") as fd:
    #     doc = xmltodict.parse(fd.read())

    try:
        start_x = np.int(simg.properties['openslide.bounds-x'])
        start_y = np.int(simg.properties['openslide.bounds-y'])
        width_x = np.int(simg.properties['openslide.bounds-width'])
        height_y = np.int(simg.properties['openslide.bounds-height'])
    except:
        start_x = 0
        start_y = 0
        width_x = np.int(simg.properties['aperio.OriginalWidth'])
        height_y = np.int(simg.properties['aperio.OriginalHeight'])
    down_rate = simg.level_downsamples[opt.lv]

    detect_json = []
    doc_out = {}
    doc_out['Annotations'] = {}
    doc_out['Annotations']['@MicronsPerPixel'] = simg.properties['openslide.mpp-x']
    doc_out['Annotations']['@Level'] = opt.lv
    doc_out['Annotations']['@DownRate'] = down_rate
    doc_out['Annotations']['@start_x'] = start_x
    doc_out['Annotations']['@start_y'] = start_y
    doc_out['Annotations']['@width_x'] = width_x
    doc_out['Annotations']['@height_y'] = height_y
    if 'leica.device-model' in simg.properties:
        doc_out['Annotations']['@Device'] = 'leica.device-model'
    else:
        doc_out['Annotations']['@Device'] = 'aperio.Filename'
    doc_out['Annotations']['Annotation'] = {}
    doc_out['Annotations']['Annotation']['@Id'] = '1'
    doc_out['Annotations']['Annotation']['@Name'] = ''
    doc_out['Annotations']['Annotation']['@ReadOnly'] = '0'
    doc_out['Annotations']['Annotation']['@LineColorReadOnly'] = '0'
    doc_out['Annotations']['Annotation']['@Incremental'] = '0'
    doc_out['Annotations']['Annotation']['@Type'] = '4'
    doc_out['Annotations']['Annotation']['@LineColor'] = '65280'
    doc_out['Annotations']['Annotation']['@Visible'] = '1'
    doc_out['Annotations']['Annotation']['@Selected'] = '1'
    doc_out['Annotations']['Annotation']['@MarkupImagePath'] = ''
    doc_out['Annotations']['Annotation']['@MacroName'] = ''
    doc_out['Annotations']['Annotation']['Attributes'] = {}
    doc_out['Annotations']['Annotation']['Attributes']['Attribute'] = {}
    doc_out['Annotations']['Annotation']['Attributes']['Attribute']['@Name'] = 'glomerulus'
    doc_out['Annotations']['Annotation']['Attributes']['Attribute']['@Id'] = '0'
    doc_out['Annotations']['Annotation']['Attributes']['Attribute']['@Value'] = ''
    doc_out['Annotations']['Annotation']['Plots'] = None
    doc_out['Annotations']['Annotation']['Regions'] = {}
    doc_out['Annotations']['Annotation']['Regions']['RegionAttributeHeaders'] = {}
    doc_out['Annotations']['Annotation']['Regions']['AttributeHeader'] = []
    doc_out['Annotations']['Annotation']['Regions']['Region'] = []

    for di in range(len(detect_all)):
        detect_one = detect_all[di]
        detect_dict = {}
        detect_dict['@Id'] = str(di + 1)
        detect_dict['@Type'] = '2'
        detect_dict['@Zoom'] = '0.5'
        detect_dict['@ImageLocation'] = ''
        detect_dict['@ImageFocus'] = '-1'
        detect_dict['@Length'] = '2909.1'
        detect_dict['@Area'] = '673460.1'
        detect_dict['@LengthMicrons'] = '727.3'
        detect_dict['@AreaMicrons'] = '42091.3'
        detect_dict['@Text'] = ('%.3f' % detect_one[3])
        detect_dict['@NegativeROA'] = '0'
        detect_dict['@InputRegionId'] = '0'
        detect_dict['@Analyze'] = '0'
        detect_dict['@DisplayId'] = str(di + 1)
        detect_dict['Attributes'] = None
        detect_dict['Vertices'] = '0'
        detect_dict['Vertices'] = {}
        detect_dict['Vertices']['Vertex'] = []

        if 'leica.device-model' in simg.properties:  #leica
            coord1 = {}
            coord1['@X'] = str(height_y - (detect_one[1] - detect_one[2]) * down_rate)
            coord1['@Y'] = str((detect_one[0] - detect_one[2]) * down_rate)
            coord1['@Z'] = '0'
            coord2 = {}
            coord2['@X'] = str(height_y - (detect_one[1] + detect_one[2]) * down_rate)  # 左右
            coord2['@Y'] = str((detect_one[0] + detect_one[2]) * down_rate)  # 上下
            coord2['@Z'] = '0'
            detect_dict['Vertices']['Vertex'].append(coord1)
            detect_dict['Vertices']['Vertex'].append(coord2)
        elif 'aperio.Filename' in simg.properties:
            coord1 = {}
            coord1['@X'] = str((detect_one[0] - detect_one[2]) * down_rate)
            coord1['@Y'] = str((detect_one[1] - detect_one[2]) * down_rate)
            coord1['@Z'] = '0'
            coord2 = {}
            coord2['@X'] = str((detect_one[0] + detect_one[2]) * down_rate)  # 左右
            coord2['@Y'] = str((detect_one[1] + detect_one[2]) * down_rate)  # 上下
            coord2['@Z'] = '0'
            detect_dict['Vertices']['Vertex'].append(coord1)
            detect_dict['Vertices']['Vertex'].append(coord2)
        doc_out['Annotations']['Annotation']['Regions']['Region'].append(detect_dict)

    out = xmltodict.unparse(doc_out, pretty=True)
    with open(xml_file, 'wb') as file:
        file.write(out.encode('utf-8'))

    os.system('rm -r "%s"' % (os.path.join(working_dir, '2d_patch')))

    return

def demo(opt):
    # opt.lv = 2
    # opt.zoom = 4
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # opt.debug = max(opt.debug, 1)


    demo_dir = opt.demo_dir

    if os.path.isdir(opt.demo):
        scn_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in scn_ext:
                scn_names.append(os.path.join(opt.demo, file_name))
    else:
        scn_names = [opt.demo]

    for (demo_scn) in scn_names:
        run_one_scn(demo_scn, demo_dir, opt)






if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
