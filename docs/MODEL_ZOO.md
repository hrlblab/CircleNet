# MODEL ZOO

### Common settings and notes

- The experiments are run with an Nvidia 1080 Ti (11 GB), PyTorch 0.4.1, CUDA 9.0, and CUDNN 7.1.

## Object Detection

### MoNuSeg reconstruction: newly trained model

The [reconstructed dataset](MONUSEG_RECONSTRUCTION.md) has a separately trained
COCO-initialized two-stack Hourglass checkpoint. These results use the new
reconstructed test set and must not be presented as historical benchmark results.

| Model | AP (0.50–0.95) | AP50 | AP75 | Download |
|---|---:|---:|---:|---|
| CircleNet-HG, reconstructed MoNuSeg | 44.15 | 81.91 | 43.28 | [Staging folder (public hosting pending)](https://drive.google.com/drive/folders/1naE0aCfiyBTolj7FCKVwUB7Sv6vxdDBl) |

The full checkpoint is named `hourglass_coco_best.pth` (765,800,105 bytes).
Upload/rehosting is pending; this staging folder is not yet a verified public
download endpoint. Verify its SHA-256 before use:

```
2744597be28071b738243ac9f3113b16b700537becde47e7646bc4f552bb0db7
```

See [training, evaluation, and provenance](MONUSEG_RECONSTRUCTION.md).
No newly validated DLA checkpoint is provided.

### Historical MoNuSeg 2018 results

The following values are retained from the paper for reference. The original
model files were reported lost; verified historical download files are not
currently available.

| Model                    |      mAP     |  Download | 
|--------------------------|-------------|----------------|
|[circledet\_monuseg\_hg](../experiments/circledet_monuseg_hg.sh)|    48.7     | Not currently verified |
|[circledet\_monuseg\_dla](../experiments/circledet_monuseg_dla.sh)  |   48.6  | Not currently verified |
|[ctdet\_monuseg\_hg](../experiments/ctdet_monuseg_hg.sh)|   44.7 | model (in progress)|
|[ctdet\_monuseg\_dla](../experiments/ctdet_monseg_dla.sh) |   39.9  | model (in progress)|

The former HG and DLA download shares are not verified MoNuSeg references.
Both currently contain epoch-10 Hourglass checkpoints. The DLA share is also
the human-kidney WSI model URL in the README, so it is not a valid MoNuSeg DLA
download. See the [checkpoint audit](MONUSEG_RECONSTRUCTION.md#existing-checkpoint-status).
