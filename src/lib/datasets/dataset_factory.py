from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.registry import LazyRegistry

dataset_factory = LazyRegistry(__package__, {
  'coco': ('.dataset.coco', 'COCO'),
  'pascal': ('.dataset.pascal', 'PascalVOC'),
  'kitti': ('.dataset.kitti', 'KITTI'),
  'coco_hp': ('.dataset.coco_hp', 'COCOHP'),
  'kidpath': ('.dataset.kidpath', 'KidPath'),
  'kidpath_old': ('.dataset.kidpath_old', 'KidPath_old'),
  'kidmouse': ('.dataset.kidmouse', 'KidMouse'),
  'kidney_first_batch_081617_ADE': ('.dataset.kidney_first_batch_081617_ADE', 'KidPath_FirstBatch_081617_ADE'),
  'kidpath_first_batch_R24': ('.dataset.kidney_first_batch_R24', 'KidPath_FirstBatch_R24'),
  'monuseg': ('.dataset.monuseg', 'MoNuSeg'),
  'nucls': ('.dataset.nucls', 'NuCLS'),
})

_sample_factory = LazyRegistry(__package__, {
  'exdet': ('.sample.exdet', 'EXDetDataset'),
  'ctdet': ('.sample.ctdet', 'CTDetDataset'),
  'ddd': ('.sample.ddd', 'DddDataset'),
  'multi_pose': ('.sample.multi_pose', 'MultiPoseDataset'),
  'circledet': ('.sample.circledet', 'CirCleDataset'),
})


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset

