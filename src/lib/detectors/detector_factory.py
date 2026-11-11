from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.registry import LazyRegistry

detector_factory = LazyRegistry(__package__, {
  'exdet': ('.exdet', 'ExdetDetector'),
  'ddd': ('.ddd', 'DddDetector'),
  'ctdet': ('.ctdet', 'CtdetDetector'),
  'multi_pose': ('.multi_pose', 'MultiPoseDetector'),
  'circledet': ('.circledet', 'CircledetDetector'),
})
