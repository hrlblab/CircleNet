from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.circledet import CirCleDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.kidpath import KidPath
from .dataset.kidpath_old import KidPath_old
from .dataset.kidmouse import KidMouse
from .dataset.kidney_first_batch_081617_ADE import KidPath_FirstBatch_081617_ADE
from .dataset.kidney_first_batch_R24 import KidPath_FirstBatch_R24

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'kidpath': KidPath,
  'kidpath_old': KidPath_old,
  'kidmouse': KidMouse,
  'kidney_first_batch_081617_ADE': KidPath_FirstBatch_081617_ADE,
  'kidpath_first_batch_R24': KidPath_FirstBatch_R24,
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'circledet': CirCleDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
