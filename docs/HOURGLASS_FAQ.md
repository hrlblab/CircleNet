# Hourglass troubleshooting and circle datasets

## `ModuleNotFoundError: external.nms` (issue #29)

Use the [modern Hourglass environment](INSTALL_HOURGLASS.md). Standard
`circledet --arch hourglass` no longer imports the optional compiled NMS or
DCNv2 extensions. Do not fix this by creating an empty `_ext` or `nms` module.
Other legacy detector tasks may still need their documented compiled modules.
For a headless server, pass `--demo_dir outputs/demo`; the headless OpenCV
package cannot display GUI windows. Native Windows has not been validated;
the reported Windows issue still needs reporter confirmation or a WSL test.

## CUDA toolkit / PyTorch mismatch and missing `_ext` (issue #26)

Print `torch.__version__`, `torch.version.cuda`, and
`torch.cuda.is_available()` in the environment running CircleNet. `nvidia-smi`
reports driver capability; it is not the CUDA toolkit selected by `nvcc`.
For the modern Hourglass path no CUDA extension is compiled: use a compatible
PyTorch wheel and NVIDIA driver. There is no `_ext` directory to create.

For legacy DLA/ResNet-DCN, check `nvcc --version` and the toolkit selected by
`CUDA_HOME` against the PyTorch build. Use the DCNv2 branch matching that
legacy PyTorch release and rebuild within the same environment, following
[INSTALL2023.md](INSTALL2023.md). Installing new package versions alone does
not port an old CUDA extension. DLA compilation is outside the validated
modern Hourglass support scope.

## What does `score` mean? Which threshold? (issue #28)

Circle output rows are `[center_x, center_y, radius, score, class]`, in pixels
of the original image. Result dictionaries are keyed by one-based category ID;
the final row's class index is zero-based. `results.json` includes
`circle_center`, `circle_radius`, `score`, `image_id`, and `category_id`.

The score is the sigmoid center-heatmap response at a decoded local maximum.
It is a ranking confidence, **not a calibrated probability that the circle is
correct**. Choose an operating threshold using labeled validation data and the
precision/recall needs of your application; there is no universal threshold.

`--vis_thresh` affects displayed/saved overlays only. It does not remove rows
from returned results or set the AP evaluation threshold. For deployment,
filter each row by `row[3] >= threshold`, and reject non-positive radii.
AP evaluates a ranking across detection scores; do not tune a display cutoff
on test data or claim that changing it improves AP.

## How do I annotate my own circular objects? (issue #27)

CircleNet detects circles; it does not produce an instance-segmentation mask.
For each object, obtain a center `(cx, cy)` and positive radius `r` in original
image pixels. You can annotate circles directly or derive an enclosing circle
from a polygon/mask. Record the conversion and treatment of truncated objects.
Different choices can change both training labels and evaluation scores.

A minimal one-class COCO-style annotation file looks like:

```json
{
  "images": [{"id": 1, "file_name": "example.png", "width": 512, "height": 512}],
  "categories": [{"id": 1, "name": "fruit"}],
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
    "bbox": [116, 148, 24, 24], "area": 452.389342,
    "circle_center": [128, 160], "circle_radius": 12
  }]
}
```

Here `bbox` is `[left, top, width, height]`, not two corner points. These
coordinates describe a circle centered at `(128,160)` with radius 12. For a
polygon-derived dataset, use the mask's actual area and bounding box and retain
segmentation if needed; the example uses the ideal circle's area and box.
Do not divide centers/radii by the model's stride: the sampler performs the
image/output-grid transformation. Object IDs must be unique and refer to valid
image/category IDs. Keep source images/patients separate across splits.

For a one-class experiment, the existing MoNuSeg loader can read these fields
with its expected `train/`, `val/`, `test/` folders and
`MoNuSeg_{split}2021.json` filenames. For a named custom dataset, copy
`src/lib/datasets/dataset/monuseg.py`, change its paths, categories, class names,
and metadata, and register it in `dataset_factory` as, for example,
`'fruit': ('.dataset.fruit', 'Fruit')`. Train with `--dataset fruit`.
The generic demo's task defaults still assume the built-in class metadata;
configure custom class metadata before constructing a detector for other
class counts. See [DEVELOP.md](DEVELOP.md).

## XML exists but ImageScope shows no annotations (issue #32)

The modern Hourglass path does not validate the SCN/WSI pipeline or ImageScope.
First confirm the XML contains regions and finite vertices:

```sh
python src/tools/check_aperio_xml.py /path/to/slide.xml --width 100000 --height 80000
```

Use the **level-0** slide dimensions, not the downsampled inference image size.
The checker reports empty regions, invalid coordinates and out-of-bounds
vertices; successful checks do not guarantee that ImageScope will render it.
Confirm the viewer has actually loaded the intended annotation file and that
its annotation layer is visible. Coordinate transforms, crop bounds/offsets,
downsampling, and Leica-versus-Aperio metadata all need checking against the
actual slide. Please provide an anonymized XML sample, slide dimensions/vendor
metadata, command and viewer version when reporting this issue. A file's
presence in the same directory is not an end-to-end rendering test.

## Missing model/data and generalization (issues #30, #31, #33)

The runtime update does not restore lost archives, publish new downloads, or
validate the former model-zoo links. Dataset/checkpoint rehosting and the
reconstruction caveats are tracked separately. In particular, the old
MoNuSeg-DLA share is not proof of a valid DLA checkpoint.

A tissue-specific pretrained model is not guaranteed to detect arbitrary
circles or fruit without fine-tuning. Evaluate on labeled target-domain data.
Code compatibility and successful checkpoint loading do not establish task
accuracy. Keep asset issues open until the actual public files are verified.
