# MoNuSeg reconstruction

Download the [three dataset splits](DATA.md) and [Hourglass checkpoint](MODEL_ZOO.md).
These are new data and weights, not the recovered historical archive.
The final splits and checkpoint are hosted on Hugging Face; rebuild scripts stay here.

## Dataset

The 200 / 100 / 140 train / validation / test patches are 512×512 RGB images.
We used 20 / 10 / 14 source cases, ten crops per case and seed `20261111`.
Cases do not overlap between splits.

Labels come from original XML polygons, rasterized and clipped to each crop.
Intersections of at least three pixels are kept; minimum enclosing circles
are fitted to foreground pixel centers. Truncated instances are marked.

Original crop coordinates, splits and boundary conventions are unknown.
The selected source files contain 16,966 training and 6,697 test nuclei,
versus the publication's 21,623 and approximately 7,000. No labels were invented.
**These differences prevent a reproduction claim.**

[Rebuild scripts and records](../tools/monuseg/) include crop coordinates,
checksums, source verification and validation results. All 440 released images
passed integrity, geometry, case-separation and COCO/CircleNet-reader checks.
Source images and XML are not needed for training.

Data use **CC BY-NC-SA 4.0**; scripts use MIT. Follow the
[MoNuSeg terms](https://monuseg.grand-challenge.org/Data/) and cite
[Kumar et al.](https://doi.org/10.1109/TMI.2019.2947628).

## Checkpoint

The original two-stack Hourglass was initialized from
[COCO CenterNet-HG](https://drive.google.com/file/d/13F904yXltGIqa_K3g3-FLleaNX0n7c9O/view).
Training used Adam, batch size 4, learning rate 0.00025, 50 epochs, full precision
and seed `20261111`; lowest validation loss selected epoch 33.

[Reported scores](MODEL_ZOO.md) use the final stack, positive radii, scores ≥ 0.01
and no test augmentation. The standard CLI keeps all candidates and reports
about 44.2 AP. This was one RTX 3090 run with modern PyTorch/CUDA, not the
paper's runtime. Rotation consistency was not reproduced.

The checkpoint was tested with Python 3.12 and PyTorch 2.14.1+cu130. The legacy
PyTorch 0.4.1 setup is not supported for this checkpoint; modern runtime fixes
are tracked in [PR #35](https://github.com/hrlblab/CircleNet/pull/35).

The former HG and DLA downloads both contain epoch-10 Hourglass checkpoints.
The DLA share is also the README's human-kidney model link; it is not a verified
MoNuSeg DLA model. The HG file's training task is unknown. No new DLA checkpoint
is supplied.
