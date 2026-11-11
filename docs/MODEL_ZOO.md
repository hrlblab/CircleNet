# MODEL ZOO

### Common settings and notes

- Historical experiments used an Nvidia 1080 Ti (11 GB), PyTorch 0.4.1, CUDA 9.0,
  and CUDNN 7.1.

## Object Detection

### Reconstructed MoNuSeg

This checkpoint was trained on the [new splits](DATA.md), not the historical
archive. See [training settings and limitations](MONUSEG_RECONSTRUCTION.md).

| Model | AP | AP50 | AP75 | Download |
|---|---:|---:|---:|---|
| CircleNet-HG | 44.15 | 81.91 | 43.28 | [model](https://drive.google.com/file/d/12k4Kze7OkSd1ItcAta92p4PCaR35w5z4/view) |

Save as `models/hourglass_coco_best.pth` (765,800,105 bytes). SHA-256:

```
2744597be28071b738243ac9f3113b16b700537becde47e7646bc4f552bb0db7
```

### Historical MoNuSeg 2018

Paper results are retained below for reference. The original model files were
reported lost. Replacement HG/DLA shares are not verified MoNuSeg models;
[no new DLA checkpoint is provided](MONUSEG_RECONSTRUCTION.md).

| Model | mAP | Download |
|---|---:|---|
|[circledet\_monuseg\_hg](../experiments/circledet_monuseg_hg.sh)|48.7|Not currently verified|
|[circledet\_monuseg\_dla](../experiments/circledet_monuseg_dla.sh)|48.6|Not currently verified|
|[ctdet\_monuseg\_hg](../experiments/ctdet_monuseg_hg.sh)|44.7|model (in progress)|
|[ctdet\_monuseg\_dla](../experiments/ctdet_monseg_dla.sh)|39.9|model (in progress)|
