# Shared runtime fixes: validation evidence

Validated on November 11, 2026 with an RTX 3090 (24 GiB), Rocky Linux 10.2,
NVIDIA driver 610.43.02, and Python 3.12.12. The test environment used PyTorch
2.14.1+cu130, NumPy 1.26.4, OpenCV headless 4.11.0.86, Pillow 12.3.0, SciPy
1.15.3, Matplotlib 3.10.9, Numba 0.68.1, pycocotools 2.0.11, bbox 0.9.2,
progress 1.6.1, and tensorboardX 2.6.5. `pip check` passed. A Python 3.10.19
control also completed demo, training/resume and evaluation. This records
the environment used for testing; it is not a new dependency profile or a
declaration of modern compatibility for all backbones.

## Checks performed

- All 17 CPU regression tests passed: lazy optional imports, custom registry
  registration and membership without imports, restricted
  checkpoint/Adam loading, legacy loading-signature compatibility, CPU
  inference, score/coordinate handling, circle suppression, headless overlays,
  a minimal custom dataset and perfect-prediction evaluation, XML checks,
  prefetch options without global state, and task-appropriate evaluation.
- The standard `src/demo.py` entry point saved a 512×512 overlay using the
  trained Hourglass checkpoint. Multiscale inference (`--test_scales 0.5,1
  --nms`) also completed without compiled NMS/DCNv2 extensions.
- The standard `src/main.py` training entry point loaded COCO Hourglass
  initialization, trained on all 200 reconstructed training patches and
  validated on all 100 validation patches for one epoch (batch size 4).
  Resuming the saved experiment completed a second epoch with Adam state.
- The standard `src/main.py --test --ontestdata` path evaluated all 140 test
  patches with the previously trained epoch-33 Hourglass checkpoint. Both
  Python versions produced the following rounded evaluator output:

| Metric | Result |
| --- | ---: |
| Circle AP, IoU 0.50:0.95 | 0.442 |
| Circle AP50 | 0.821 |
| Circle AP75 | 0.433 |
| Circle AP small | 0.455 |
| Circle AP medium | 0.309 |
| Circle AR, maxDets=1000 | 0.540 |

During review, both `src/test.py --ontestdata` loading modes (default prefetch
and `--not_prefetch_test`) also evaluated all 140 test patches and matched
every rounded metric above.

The original circle evaluator and its two-decimal result serialization were
retained. This standard evaluation keeps up to 1,000 detections; it differs
slightly from earlier custom-driver results that prefiltered low scores.

## Interpretation and limits

These checks establish compatibility of the standard Hourglass commands with
the tested modern environment. The smoke training used the default learning
rate (0.000125); it is not a new full-length training experiment. Evaluation
used an existing checkpoint from a single 50-epoch, COCO-initialized run with
batch size 4 and learning rate 0.00025, selected by validation loss at epoch 33.

The reconstructed dataset uses new crops and splits. It is not the recovered
historical archive, and its source annotation counts differ from publication
counts. These numbers do not reproduce the paper's benchmark. Checkpoint and
dataset publication is separate from these shared runtime fixes.

Legacy installation instructions and requirements remain available. Legacy
PyTorch itself, DLA/DCNv2, native Windows, multi-GPU execution, and the
SCN/ImageScope pipeline were not validated by this work. The XML tool checks
structure and coordinates, not viewer rendering. See the
[FAQ](RUNTIME_FAQ.md) for the corresponding open support questions.
