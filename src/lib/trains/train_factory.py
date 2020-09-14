from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .circledet import CircleTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'circledet': CircleTrainer,
  'multi_pose': MultiPoseTrainer, 
}
