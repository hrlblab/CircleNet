# Modern Hourglass installation

This is a separate, tested path for **single-GPU CircleNet Hourglass circle
detection on Linux**. The historical [requirements](../requirements.txt),
[PyTorch 0.4.1 setup](INSTALL.md), and [2023 setup](INSTALL2023.md) remain
available for legacy work. Do not combine the environments or edit PyTorch's
installed source files for this modern path.

## Environment

Verified environment: Python 3.12.12, PyTorch 2.14.1 (CUDA runtime 13.0), NumPy 1.26.4,
OpenCV headless 4.11.0.86, Rocky Linux 10.2, RTX 3090 24 GiB, NVIDIA driver
610.43.02. Other platforms, native Windows, multi-GPU, and other architectures
have not been validated by these checks. WSL/Linux is the suggested route for
Windows users; native-Windows compatibility is not claimed.

Use maintained Python 3.12 for new environments. Python 3.10 was used for the
earlier reconstruction experiment but reached end of life in October 2026.
See the [Python support schedule](https://devguide.python.org/versions/).

From the repository root, using Python 3.12:

```sh
python3.12 -m venv .venv-hourglass
source .venv-hourglass/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hourglass.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Install a PyTorch wheel compatible with your OS/GPU driver using the
[official installation guide](https://pytorch.org/get-started/locally/) if your
hardware requires another build. That build is not automatically covered by
the tested version list. The pins intentionally retain NumPy 1.26 for the
legacy `bbox` package rather than claiming general NumPy 2 support.

Hourglass does **not** require compiling DCNv2 or `external.nms`. Factories
import only the selected dataset/task/backbone. Optional circle soft-NMS uses
NumPy circle geometry rather than the box-oriented Cython routine. DLA and
ResNet-DCN still require their legacy extension and have not been ported here.

For GUI windows, replace `opencv-python-headless` with `opencv-python` of the
same version in this environment; do not install both. The tested headless
commands save images and do not require a display server.

## Data and checkpoints

Place a compatible dataset at `data/monuseg`, following [DATA.md](DATA.md).
MoNuSeg release/rehosting work is separate from this runtime update. Do not
assume any available reconstruction has the paper's original crops or labels.
Use a verified Hourglass checkpoint with circle heads. A filename or model-zoo
label alone does not establish architecture or training provenance.

`load_model` uses restricted `weights_only=True` loading when supported by
PyTorch, with no automatic unsafe-pickle fallback. Historical PyTorch versions
without that option keep their original loading behavior. Load only trusted
checkpoints. COCO initialization adapts the class heatmap and adds the circle
radius heads; task-specific skipped/new heads are printed by the loader.

## Standard commands

These use the repository entry points, not an alternate training driver.
Run from the repository root; replace checkpoint paths as appropriate.

```sh
# Headless demo: save the overlay instead of opening an OpenCV GUI.
OMP_NUM_THREADS=6 python src/demo.py circledet --arch hourglass \
  --load_model models/hourglass_coco_best.pth --demo docs/demo.png \
  --demo_dir outputs/demo

# Train for one full epoch (including validation) as an environment smoke test.
OMP_NUM_THREADS=6 python src/main.py circledet --arch hourglass --dataset monuseg \
  --exp_id hourglass_smoke --batch_size 4 --num_workers 2 --num_epochs 1 \
  --load_model models/ctdet_coco_hg.pth --print_iter 25

# Evaluate all test patches with a trained circle checkpoint.
OMP_NUM_THREADS=6 python src/main.py circledet --arch hourglass --dataset monuseg \
  --exp_id hourglass_eval --load_model models/hourglass_coco_best.pth \
  --test --ontestdata --print_iter 50

# Continue a saved experiment, including its Adam state and learning-rate schedule.
OMP_NUM_THREADS=6 python src/main.py circledet --arch hourglass --dataset monuseg \
  --exp_id hourglass_smoke --batch_size 4 --num_workers 2 --num_epochs 2 --resume

# Small CPU regression tests: imports, checkpoints, geometry, data, and evaluation.
python -m unittest discover -s tests -v
```

The smoke epoch establishes optimizer/loader/checkpoint compatibility; it is not
a replacement for the full training protocol or a benchmark result.

## Related support questions

See [validation results and limits](HOURGLASS_VALIDATION.md) for the tested
demo, training/resume, and complete test evaluation on an RTX 3090.

See [the troubleshooting and dataset FAQ](HOURGLASS_FAQ.md) for missing NMS,
CUDA mismatch, scores/thresholds, custom circle annotations, and ImageScope XML.
