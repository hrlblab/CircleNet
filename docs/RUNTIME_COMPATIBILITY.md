# Runtime compatibility and shared fixes

CircleNet keeps its existing architectures and installation instructions:
[original setup](INSTALL.md), [2023 setup](INSTALL2023.md), and
[requirements.txt](../requirements.txt). This change does not introduce a
backbone-specific dependency profile or replace the project's dependencies.

## Shared behavior

Task, dataset, trainer and backbone factories import dependencies when their
entries are selected. Selecting one component no longer requires installing
unrelated components' extensions. DLA/ResNet-DCN still require their DCNv2
extension; other legacy detector tasks may require compiled box NMS.

Circle detection uses NumPy circle soft-NMS rather than passing circle rows
to the box-oriented compiled routine. Multiscale postprocessing preserves
confidence scores and class IDs, and detection limits rank confidence scores.
These fixes apply to the circle detector regardless of its backbone.

`load_model` uses restricted `weights_only=True` loading when PyTorch supports
it, without an automatic unsafe-pickle fallback. Older versions without that
option retain their loading behavior. Load only trusted checkpoints. Shape
differences during transfer learning are reported by the existing loader.

CPU timing skips CUDA synchronization. Image demos can save overlays with
`--demo_dir /path/to/output` instead of opening GUI windows. This does not
change the webcam/video GUI path. `src/test.py` routes circle predictions to
circle AP and box predictions to box AP; its prefetch loader uses its own
options, and `--ontestdata` selects the test split in both loading modes.

## Dependency troubleshooting

Check `torch.__version__`, `torch.version.cuda` and
`torch.cuda.is_available()` in the actual Python environment. The driver
capability reported by `nvidia-smi` is not the toolkit selected by `nvcc`.
If compiling DCNv2, check `CUDA_HOME` and `nvcc --version` against the PyTorch
build and follow the matching legacy instructions. Installing newer Python
packages does not port an old CUDA extension.

Use either `opencv-python` (GUI) or `opencv-python-headless`, not both. The
headless package cannot open OpenCV windows. NumPy 2 compatibility is not
claimed; the validation environment used NumPy 1.26.4 with `bbox` 0.9.2.

## Verification scope

Run the shared regression tests from the repository root:

```sh
python -m unittest discover -s tests -v
```

See [validation evidence](RUNTIME_VALIDATION.md) for the precise environment
and GPU results. The end-to-end checks used Hourglass because a verified
checkpoint and training initialization were available. This is evidence for
that configuration, not a promise that all backbones run on modern PyTorch.
Legacy PyTorch, DLA/DCNv2, native Windows, multi-GPU and SCN/ImageScope were
not validated. See the [FAQ](RUNTIME_FAQ.md) for issue-specific guidance.
