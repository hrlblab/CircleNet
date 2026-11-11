from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.registry import LazyRegistry

train_factory = LazyRegistry(__package__, {
  'exdet': ('.exdet', 'ExdetTrainer'),
  'ddd': ('.ddd', 'DddTrainer'),
  'ctdet': ('.ctdet', 'CtdetTrainer'),
  'circledet': ('.circledet', 'CircleTrainer'),
  'multi_pose': ('.multi_pose', 'MultiPoseTrainer'),
})
