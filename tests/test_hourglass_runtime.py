"""CPU regression tests; GPU entry-point checks are documented separately."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/lib')]
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import torch
from PIL import Image
from detectors.circledet import CircledetDetector
from models.model import create_model, load_model, save_model
from utils.circle_nms import circle_iou, soft_nms_circles


class RuntimeTests(unittest.TestCase):
    def test_standard_entrypoints_do_not_import_optional_extensions(self):
        code = '''
import sys
import demo, main
from detectors.detector_factory import detector_factory
from trains.train_factory import train_factory
from datasets.dataset_factory import get_dataset
assert detector_factory['circledet'].__name__ == 'CircledetDetector'
assert train_factory['circledet'].__name__ == 'CircleTrainer'
get_dataset('monuseg', 'circledet')
assert 'external.nms' not in sys.modules
assert not any('DCNv2' in name for name in sys.modules)
assert 'torchvision' not in sys.modules
assert 'openslide' not in sys.modules
'''
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path[:2]))
        subprocess.run([sys.executable, '-c', code], env=env, check=True)

    def test_checkpoint_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pth'
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            save_model(str(path), 3, model, optimizer)
            restored = torch.nn.Linear(2, 1)
            optim = torch.optim.Adam(restored.parameters())
            _, optim, epoch = load_model(restored, str(path), optim, True, .001, [2])
            self.assertEqual(epoch, 3)
            self.assertAlmostEqual(optim.param_groups[0]['lr'], .0001)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b)

    def test_loader_retains_the_legacy_torch_load_signature(self):
        original_load = torch.load
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pth'
            model = torch.nn.Linear(2, 1)
            save_model(str(path), 1, model)
            def legacy_load(path, map_location=None):
                return original_load(path, map_location=map_location, weights_only=True)
            with patch('torch.load', legacy_load):
                load_model(torch.nn.Linear(2, 1), str(path))

    def test_dcn_architecture_reports_its_optional_dependency(self):
        with self.assertRaisesRegex(ImportError, 'legacy DCNv2'):
            create_model('dla_34', {'hm': 1, 'cl': 1, 'reg': 2}, 256)

    def test_circle_postprocessing_does_not_scale_score_or_class(self):
        detector = CircledetDetector.__new__(CircledetDetector)
        detector.num_classes = 1
        detector.opt = Mock(num_classes=1)
        raw = torch.tensor([[[64., 64., 4., .8, 0.]]])
        meta = dict(c=np.array([256., 256.]), s=512., out_height=128, out_width=128)
        result = detector.post_process(raw, meta, scale=2)[1][0]
        np.testing.assert_allclose(result, [128, 128, 8, .8, 0], atol=1e-6)
        meta['s'] = np.array([512., 512.])
        np.testing.assert_allclose(detector.post_process(raw, meta, 2)[1][0], result)

    def test_result_limit_uses_confidence_not_class_id(self):
        detector = CircledetDetector.__new__(CircledetDetector)
        detector.num_classes = 1
        detector.max_per_image = 1
        detector.scales = [1]
        detector.opt = Mock(nms=False)
        detections = {1: np.array([[10, 10, 4, .2, 0], [20, 20, 4, .9, 0]], np.float32)}
        result = detector.merge_outputs([detections])[1]
        self.assertEqual(result.shape, (1, 5))
        self.assertAlmostEqual(float(result[0, 3]), .9, places=6)

    def test_circle_soft_nms_preserves_geometry_and_class(self):
        dets = np.array([[0, 0, 2, .9, 3], [0, 0, 2, .8, 3], [20, 20, 2, .7, 3]], np.float32)
        result = soft_nms_circles(dets)
        np.testing.assert_allclose(result[:, 3], [.9, .7, .8*np.exp(-2)], rtol=1e-6)
        np.testing.assert_array_equal(result[:, 4], [3, 3, 3])
        np.testing.assert_array_equal(result[:, 2], [2, 2, 2])
        np.testing.assert_allclose(dets[:, 3], [.9, .8, .7])
        self.assertEqual(soft_nms_circles(np.empty((0, 5), np.float32)).shape, (0, 5))

    def test_circle_iou_boundary_cases(self):
        ref = np.array([0., 0., 2.])
        other = np.array([[0, 0, 2], [0, 0, 1], [4, 0, 2], [0, 0, 0]], float)
        np.testing.assert_allclose(circle_iou(ref, other), [1, .25, 0, 0])

    def test_circle_iou_matches_original_geometry(self):
        from datasets.eval_protocals.mask import circleIOU
        rng = np.random.RandomState(17)
        circle = np.array([10., 10., 8.])
        others = np.column_stack([rng.uniform(0, 20, (20, 2)), rng.uniform(1, 10, 20)])
        np.testing.assert_allclose(circle_iou(circle, others), circleIOU([circle], others)[0], atol=1e-6)

    def test_xml_checker_distinguishes_empty_and_valid_annotations(self):
        from tools.check_aperio_xml import check_annotations
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.xml'
            path.write_text('<Annotations/>')
            self.assertTrue(check_annotations(path)['errors'])
            path.write_text('<Annotations><Annotation><Regions><Region><Vertices>'
                            '<Vertex X="10" Y="20"/><Vertex X="30" Y="40"/>'
                            '</Vertices></Region></Regions></Annotation></Annotations>')
            self.assertFalse(check_annotations(path, 100, 100)['errors'])
            self.assertFalse(check_annotations(path, 100, 100)['warnings'])
            self.assertTrue(check_annotations(path, 25, 100)['warnings'])

    def test_cpu_circle_decoder_does_not_synchronize_cuda(self):
        detector = CircledetDetector.__new__(CircledetDetector)
        detector.opt = Mock(reg_offset=True, flip_test=False, K=1)
        detector.model = lambda image: [dict(hm=torch.zeros(1, 1, 4, 4),
                                            cl=torch.ones(1, 1, 4, 4),
                                            reg=torch.zeros(1, 2, 4, 4))]
        with patch('torch.cuda.synchronize', side_effect=AssertionError('CPU called CUDA')):
            _, detections = detector.process(torch.zeros(1, 3, 16, 16))
        self.assertEqual(tuple(detections.shape), (1, 1, 5))

    def test_headless_render_does_not_filter_returned_detections(self):
        detector = CircledetDetector.__new__(CircledetDetector)
        detector.num_classes = 1
        detector.pause = False
        detector.image_name = 'sample.png'
        with tempfile.TemporaryDirectory() as folder:
            detector.opt = Mock(filter_boarder=False, vis_thresh=.3, demo_dir=folder)
            debugger = Mock()
            results = {1: np.array([[10, 10, 3, .2, 0], [30, 30, 4, .8, 0]])}
            detector.show_results(debugger, np.zeros((64, 64, 3)), results)
            self.assertEqual(debugger.add_coco_circle.call_count, 1)
            debugger.save_all_imgs.assert_called_once_with(folder)
            debugger.show_all_imgs.assert_not_called()
            self.assertEqual(len(results[1]), 2)

    def test_result_limit_also_caps_tied_scores(self):
        detector = CircledetDetector.__new__(CircledetDetector)
        detector.num_classes = 1
        detector.max_per_image = 1
        detector.scales = [1]
        detector.opt = Mock(nms=False)
        dets = {1: np.array([[10, 10, 4, .9, 0], [20, 20, 4, .9, 0]], np.float32)}
        self.assertEqual(len(detector.merge_outputs([dets])[1]), 1)

    def test_minimal_custom_circle_sample_and_perfect_evaluation(self):
        from opts import opts
        from datasets.dataset_factory import get_dataset
        from datasets.eval_protocals.circle_eval import CIRCLEeval
        Dataset = get_dataset('monuseg', 'circledet')
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / 'monuseg'
            (data / 'val').mkdir(parents=True)
            Image.new('RGB', (512, 512), 'white').save(data / 'val/example.png')
            doc = dict(images=[dict(id=1, file_name='example.png', width=512, height=512)],
                       categories=[dict(id=1, name='nucleus')],
                       annotations=[dict(id=1, image_id=1, category_id=1, iscrowd=0,
                                         bbox=[116, 148, 24, 24], area=float(np.pi*144),
                                         circle_center=[128, 160], circle_radius=12)])
            (data / 'MoNuSeg_val2021.json').write_text(json.dumps(doc))
            opt = opts().parse(['circledet', '--arch', 'hourglass', '--gpus', '-1'])
            opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
            opt.data_dir = folder
            dataset = Dataset(opt, 'val')
            sample = dataset[0]
            self.assertEqual(sample['input'].shape, (3, 512, 512))
            self.assertEqual(int(sample['reg_mask'].sum()), 1)
            self.assertAlmostEqual(float(sample['cl'][0, 0]), 3)
            prediction = [dict(image_id=1, category_id=1, circle_center=[128, 160], circle_radius=12, score=.9)]
            evaluator = CIRCLEeval(dataset.circle, dataset.circle.loadRes(prediction), 'circle')
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            self.assertAlmostEqual(evaluator.stats[0], 1)


if __name__ == '__main__':
    unittest.main()
