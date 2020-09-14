from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import copy
import torch
import numpy as np
import math
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
# from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
import fractions

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  if opt.ontestdata:
      val_str = 'test'
  else:
      val_str = 'val'

  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, val_str),
      batch_size=1,
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    if not opt.rotate_reproduce:
        if opt.task == 'circledet':
            val_loader.dataset.run_circle_eval(preds, opt.save_dir)
        else:
            val_loader.dataset.run_eval(preds, opt.save_dir)
    else:
        opt.rotate = opt.rotate_reproduce
        val_loader = torch.utils.data.DataLoader(
            Dataset(opt, val_str),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        _, preds2 = trainer.val(0, val_loader)
        preds2_rot = copy.deepcopy(preds2)
        if opt.task == 'circledet':
            preds2_rot = correct_rotate_circle(preds2_rot, 512, 512, 90)
            all_box, match_box = caculate_matching_rate_circle(preds2_rot,preds, 0.5)
            all_box2, match_box2 = caculate_matching_rate_circle(preds, preds2_rot, 0.5)
            print(match_box*2/(all_box+all_box2))

        else:
            preds2_rot = correct_rotate(preds2_rot, 512, 512, 90)
            all_box, match_box = caculate_matching_rate(preds2_rot, preds, 0.5)
            all_box2, match_box2 = caculate_matching_rate(preds, preds2_rot, 0.5)
            print(match_box * 2 / (all_box + all_box2))
    return


  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

def correct_rotate(preds2, height, weith, rotate_degree):
    for pred in preds2:
        bboxs = preds2[pred][1]
        for bi in range(len(bboxs)):
            if rotate_degree == 90:
                x1_new = bboxs[bi][1]
                y1_new = weith - bboxs[bi][2]
                x2_new = bboxs[bi][3]
                y2_new = weith - bboxs[bi][0]
                score = bboxs[bi][4]
                bboxs[bi] = [x1_new, y1_new, x2_new, y2_new, score]
        preds2[pred][1] = bboxs

    return preds2

def correct_rotate_circle(preds2, height, weith, rotate_degree):
    for pred in preds2:
        bboxs = preds2[pred][1]
        for bi in range(len(bboxs)):
            if rotate_degree == 90:
                x1_new = bboxs[bi][1]
                y1_new = weith - bboxs[bi][0]
                radius = bboxs[bi][2]
                score = bboxs[bi][3]
                zeroval = bboxs[bi][4]
                bboxs[bi] = [x1_new, y1_new, radius, score, zeroval]
        preds2[pred][1] = bboxs

    return preds2

def caculate_matching_rate_circle(preds, preds2, thres):
    all_box = 0
    match_box = 0
    for pred in preds:
        pred_bboxs = preds[pred][1]
        pred2_bboxs = preds2[pred][1]
        for bi in range(len(pred_bboxs)):
            if pred_bboxs[bi][3]>=thres:
                all_box = all_box+1
            else:
                continue
            done_box = 0
            for bj in range(len(pred2_bboxs)):
                if pred2_bboxs[bj][3] < thres or done_box==1:
                    continue
                overlap = circleIOU([pred2_bboxs[bj]], [pred_bboxs[bi]])
                if overlap>0.5:
                    match_box = match_box+1
                    done_box = 1
    return all_box, match_box

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

def caculate_matching_rate(preds, preds2, thres):
    all_box = 0
    match_box = 0
    for pred in preds:
        pred_bboxs = preds[pred][1]
        pred2_bboxs = preds2[pred][1]
        for bi in range(len(pred_bboxs)):
            if pred_bboxs[bi][4]>=thres:
                all_box = all_box+1
            else:
                continue
            done_box = 0
            for bj in range(len(pred2_bboxs)):
                if pred2_bboxs[bj][4] < thres or done_box==1:
                    continue
                overlap = IOU(pred2_bboxs[bj], pred_bboxs[bi])
                if overlap>0.5:
                    match_box = match_box+1
                    done_box = 1
    return all_box, match_box

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

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'

  opt = opts().parse()
  main(opt)