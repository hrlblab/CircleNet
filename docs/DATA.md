# Dataset preparation

Prepare the dataset before training or evaluation. The reconstruction below is
intended for new experiments; it does not recover the historical benchmark.


### MoNuSeg 2018

- The historical preprocessed archive is unavailable. A new XML-derived
  reconstruction contains **200 training, 100 validation, and 140 test patches**.
- [Reconstruction staging folder](https://drive.google.com/drive/folders/1naE0aCfiyBTolj7FCKVwUB7Sv6vxdDBl):
  public hosting is pending; access may be restricted. See the
  [release notes and limitations](MONUSEG_RECONSTRUCTION.md) before use.
- Download the seven `MoNuSeg_*.zip` files described in the release notes and
  extract them into the same directory. Place the resulting `monuseg` directory
  under `data`:

  ~~~
  ${CircleNet_ROOT}
  |-- data
  '-- |-- monuseg
  ~~~

All patches are 512×512 RGB PNGs. Annotation files are named
`MoNuSeg_train2021.json`, `MoNuSeg_val2021.json`, and `MoNuSeg_test2021.json`,
with COCO annotations plus `circle_center` and `circle_radius`.
The package includes original source XML, lossless source image conversions,
the crop manifest, rebuild scripts, validation results, and attribution.

**These are new crops, splits, and rasterized labels.** The historical crop
coordinates, split assignment, and boundary conventions were not recovered.
The source-annotation counts also differ from the publication; see the release
notes. Do not use this package to claim reproduction of the paper's scores.
The dataset and its transformation are distributed under **CC BY-NC-SA 4.0**.

### Glomeruli Dataset

Unfortunately, the glomeruli dataset is not publicly available. Please see MoNuSeg 2018 (above) for reference.
