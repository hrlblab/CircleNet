# MoNuSeg reconstruction

For training, download the [three final splits](../../docs/DATA.md).
These scripts are only needed to rebuild or check them.

## Rebuild

Extract the three split ZIPs together. Download the original
[training](https://drive.google.com/file/d/1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA/view)
and [test](https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view)
archives. From the CircleNet root, in a separate generation environment:

```sh
python -m pip install -r tools/monuseg/requirements.txt
python tools/monuseg/prepare_sources.py --train /path/to/original-train.zip --test /path/to/original-test.zip --output /path/to/MoNuSeg_CircleNet_Reconstructed
python tools/monuseg/rebuild_from_originals.py --root /path/to/MoNuSeg_CircleNet_Reconstructed
```

Preparation checks archive hashes and converts the selected TIFF/XML pairs.
Rebuilding overwrites the crops and labels with the recorded plan.
Generation requirements are separate from training. A Pillow 12.3 rebuild
matched all released image pixels and labels, but PNG encoding/hashes differed.

## Validate

In a CircleNet environment with its dataset-reader dependencies installed:

```sh
# Downloads only: skip this copy after rebuilding, which writes its own manifest.
cp tools/monuseg/manifest.json /path/to/MoNuSeg_CircleNet_Reconstructed/monuseg/
python tools/monuseg/validate.py --dataset /path/to/MoNuSeg_CircleNet_Reconstructed/monuseg --repo .
```

This checks image hashes, dimensions, case separation, labels and dataset readers.
It samples mask/circle enclosure; it does not run a model.

## Records

- `manifest.json`: crop coordinates, case assignments, seed and image hashes.
- `original_archive_verification.json` and `validation.json`: recorded checks.
- `canonical_train_patients.json`: selected training-source cases.
- `rebuild.py`: initial Parquet crop-plan generator, kept for provenance.
  Its mirror-mask labels were superseded by `rebuild_from_originals.py`.

These are new crops and splits, not the historical archive; see
[limitations and attribution](../../docs/MONUSEG_RECONSTRUCTION.md).
Scripts use MIT ([LICENSE](LICENSE)); data remain **CC BY-NC-SA 4.0**.
