# MoNuSeg reconstruction and new Hourglass checkpoint

## Release status

[Dataset and checkpoint staging folder](https://drive.google.com/drive/folders/1naE0aCfiyBTolj7FCKVwUB7Sv6vxdDBl).
The dataset archives are staged there. The full checkpoint upload and a verified
public download endpoint are pending; the folder currently requires access.
This draft must not be merged as a working public-download fix until the final
links have been substituted and tested without authentication.

This is a **new reconstruction**, not recovery of the historical Box archive.
Original crop coordinates, source-case split assignment, random seed, and
boundary conventions were not recovered. Neither these data nor the new
checkpoint reproduce the original benchmark by definition.

## Dataset contents and installation

Download all seven archives and extract them into the same directory:

- `MoNuSeg_train.zip`
- `MoNuSeg_val.zip`
- `MoNuSeg_test.zip`
- `MoNuSeg_source_train_1.zip`
- `MoNuSeg_source_train_2.zip`
- `MoNuSeg_source_test.zip`
- `MoNuSeg_docs_and_scripts.zip`

They assemble `MoNuSeg_CircleNet_Reconstructed/`. Copy its `monuseg` folder
to `CircleNet/data/monuseg/`. Retain the rest of the package for provenance.

| Split | 512×512 RGB PNG patches | Source cases | Patch annotations |
|---|---:|---:|---:|
| Train | 200 | 20 | 29,946 |
| Validation | 100 | 10 | 20,050 |
| Test | 140 | 14 | 19,885 |

Cases do not overlap between splits. Patch counts of annotations include
repeated nuclei in overlapping crops; they are not unique source-nucleus counts.
Ten crops are sampled per source case, with NumPy seed 20261111. Coordinates,
case assignments, and PNG SHA-256 values are recorded in `monuseg/manifest.json`.
The released manifest SHA-256 is
`eb4047ec7d7841881dcc083055de6a35952973462e801a08c4c75af350c67a80`.

Included source data comprise 44 original XML files and lossless RGB PNG
conversions of the corresponding TIFFs. Source images were checked against the
MoNuSeg mirror, while final labels were regenerated from original XML rather
than mirror masks. Polygons are rasterized with Pillow and clipped to crops;
intersections with at least three foreground pixels are kept. Border-truncated
instances are marked. Minimum enclosing circles are fitted to mask pixel
centers, which may differ from fitting continuous XML vertices.

**Source discrepancy:** the selected original files contain 16,966 training
nuclei and 6,697 test nuclei, whereas the publication reports 21,623 training
nuclei and approximately 7,000 test nuclei. No missing labels were invented.
The source release must not be assumed identical to the historical benchmark.

The data/transformed dataset are **CC BY-NC-SA 4.0**. Preserve attribution,
noncommercial use, and share-alike terms. The package includes source citations
and separate code licenses. Original sources:

- [MoNuSeg](https://monuseg.grand-challenge.org/Data/)
- [Training archive](https://drive.google.com/file/d/1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA/view)
- [Test archive](https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view)
- [RationAI mirror](https://huggingface.co/datasets/RationAI/MoNuSeg)

## Validation and rebuilding

The package's `validate.py` passed image integrity and checksum checks,
source-case split isolation, label geometry, COCO indexing, and the actual
CircleNet CIRCLE reader. Circle enclosure is sampled across masks.
The original two-stack Hourglass also passed a real-batch forward, loss,
backward, and optimizer-step test with finite outputs and gradients.
These checks demonstrate usability, not historical equivalence or complete
annotation accuracy.

From the extracted package, in an environment with its listed requirements:

```sh
python rebuild_from_originals.py --root .
python validate.py --dataset monuseg --repo /path/to/CircleNet
```

The generation dependencies and legacy training dependencies may require
separate environments.

## New checkpoint

Download `hourglass_coco_best.pth` from the staging/final release location and
place it in `CircleNet/models/`. It is 765,800,105 bytes; SHA-256:

```
2744597be28071b738243ac9f3113b16b700537becde47e7646bc4f552bb0db7
```

Training used CircleNet commit `034afbca75e7b542f29e2cfdf2f7fac393137a5a`:

- Original two-stack Hourglass with circle heatmap/radius/offset heads.
- [Official COCO CenterNet-HG initialization](https://drive.google.com/file/d/13F904yXltGIqa_K3g3-FLleaNX0n7c9O/view),
  SHA-256 `db8c6c377d08f1044f6b572a9d0b277b4ee50e9f830283d85bae1afea716c26e`.
- 972 tensors transferred; one-class heatmap outputs and radius heads initialized
  fresh. COCO box-size heads discarded.
- Adam, batch size 4, constant learning rate 0.00025, 50 epochs, full precision,
  no gradient clipping, seed 20261111. Original crop/flip/color augmentation.
- RTX 3090 with modern PyTorch/CUDA, rather than the paper's historical runtime.
- Lowest validation loss selected epoch **33** (validation loss **0.9063**).

The original sampler, network, CircleLoss, decoder, and CIRCLEeval were used.
The driver imports Hourglass directly to avoid unrelated legacy DCNv2 imports,
excludes variable-length plotting metadata from collation, and supplies legacy
NumPy scalar aliases. No network or annotation geometry was replaced.
Mixed-precision training was not used after an initial overflow in a smoke test.

Evaluation uses the selected model's final stack, no test augmentation, up to
1,000 candidates per image, positive radii, and scores >= 0.01. AP averages
circle-IoU thresholds 0.50–0.95. All values below are percentages:

| Model / dataset | AP | AP50 | AP75 |
|---|---:|---:|---:|
| New Hourglass / reconstructed validation | 40.40 | 75.94 | 38.88 |
| New Hourglass / reconstructed test | 44.15 | 81.91 | 43.28 |
| ResNet-18 baseline / reconstructed test | 26.70 | 69.10 | 14.40 |
| Paper Hourglass / historical test | 48.70 | 85.60 | 50.90 |

The numerical AP gap to the paper is 4.55 points, but different crops, splits,
labels, and runtime preclude a controlled reproduction claim. These are single
runs, not multi-seed estimates. The earlier ResNet test score was already known
when the Hourglass protocol was fixed; checkpoint selection still used only
validation loss. No rotation-consistency result is claimed for the new model.

For an image demo, from `src/`:

```sh
python demo.py circledet --arch hourglass --demo ../docs/demo.png --load_model ../models/hourglass_coco_best.pth
```

For evaluation in a compatible CircleNet environment:

```sh
python main.py circledet --exp_id reconstructed_hg --arch hourglass --batch_size 4 --master_batch 4 --dataset monuseg --load_model ../models/hourglass_coco_best.pth --test --ontestdata
```

## Existing checkpoint status

The former model-zoo HG and DLA links currently supply distinct epoch-10
Hourglass checkpoints, both loading strictly into Hourglass. The DLA URL is
also the README's human-kidney WSI-model URL and is not a verified MoNuSeg DLA
model. The two files scored 0.41% and 0.48% circle AP on reconstructed test data.
Channel-order/normalization and first/final-stack probes did not restore normal
nuclei detection on the supplied demo and fixed examples. Their exact training
datasets are not established; the HG file's task must not be inferred solely
from poor performance.

The earlier historical links return 404, and the original model files were
already reported lost. Historical scores remain references to
[Table III of the paper](https://arxiv.org/html/2110.12093#S5.T3), not promises
for either the reconstructed release or the old replacement shares.
