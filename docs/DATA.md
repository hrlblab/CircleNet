# Dataset preparation

### MoNuSeg 2018

Download the reconstructed splits:

- [Training: 200 images](https://drive.google.com/file/d/15lqGk_r-E7XuLx0AMrE4jxPBuzBZ4ogR/view)
- [Validation: 100 images](https://drive.google.com/file/d/1Dq5hAjZV7qOeUmrxZQAW2fJXnsW8pxFj/view)
- [Test: 140 images](https://drive.google.com/file/d/1bis63BTjGg17IRuqTGnD0qXnNX41CwV6/view)

Extract the three ZIPs into the same directory, then copy
`MoNuSeg_CircleNet_Reconstructed/monuseg` into `data/`:

~~~
${CircleNet_ROOT}
|-- data
'-- |-- monuseg
        |-- train
        |-- val
        |-- test
        |-- MoNuSeg_train2021.json
        |-- MoNuSeg_val2021.json
        '-- MoNuSeg_test2021.json
~~~

All images are 512×512 RGB PNGs. The JSON files contain COCO annotations plus
`circle_center` and `circle_radius`.

The original preprocessed archive is unavailable. These are new crops, splits
and XML-derived labels, so they do not reproduce the paper's benchmark.
See [reconstruction notes](MONUSEG_RECONSTRUCTION.md) and
[rebuild scripts](../tools/monuseg/). Data use **CC BY-NC-SA 4.0**.

### Glomeruli Dataset

Unfortunately, the glomeruli dataset is not publicly available. Please see MoNuSeg 2018 (above) for reference.
