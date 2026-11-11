# Getting Started

This document provides tutorials to train and evaluate CircleNet. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).

## Reconstructed MoNuSeg

Follow [dataset setup](DATA.md), then save the [checkpoint](MODEL_ZOO.md) as
`models/hourglass_coco_best.pth`. Use a compatible modern PyTorch environment;
runtime fixes are tracked separately in [PR #35](https://github.com/hrlblab/CircleNet/pull/35).
From `src/`, evaluate the test split with:

~~~
python main.py circledet --arch hourglass --dataset monuseg --load_model ../models/hourglass_coco_best.pth --test --ontestdata
~~~

See [reconstruction notes](MONUSEG_RECONSTRUCTION.md) for the training settings
and limits. These results do not reproduce the historical benchmark.

## Historical benchmark evaluation

First, download the models you want to evaluate from our [model zoo](MODEL_ZOO.md) and put them in `CircleNet_ROOT/models/`. 

### MoNuSeg 2018

The scores below are historical paper results. For the new data and checkpoint,
see [reconstruction notes](MONUSEG_RECONSTRUCTION.md); the scores are not interchangeable.

#### HG
To evaluate MoNuSeg object detection with HG, run

~~~
python main.py circledet --exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4 --dataset monuseg --load_model ../models/circledet_monuseg_hg.pth --test --ontestdata --debug 4
~~~

The paper reported an AP of `48.7` for its historical data and model.

To evaluate the rotation consistency, run 

~~~
python main.py circledet --exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model ../models/circledet_monuseg_hg.pth --test --ontestdata --rotate_reproduce 90
~~~

The historical rotation result was `0.8918255489424661`. Outputs are saved
under `exp/circledet/CircleNet_HG_Reproduce/debug`.

#### DLA

To test with DLA, run

~~~
python main.py circledet --exp_id CircleNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model ../models/circledet_monuseg_dla.pth --test --ontestdata --debug 4
~~~

The paper reported AP `48.6`; no newly validated MoNuSeg DLA checkpoint is provided.

Append `--rotate_reproduce 90` for rotation consistency. It has not been reproduced on the new data.

## Training

All the training scripts can be found in [experiments](../experiments) folder.
The experiment names correspond to the model name in the [model zoo](MODEL_ZOO.md).

By default, pytorch evenly splits the total batch size to each GPUs.
`--master_batch` allows using different batchsize for the master GPU, which usually costs more memory than other GPUs.

If the training is terminated before finishing, you can use the same commond with `--resume` to resume training. It will found the latest model with the same `exp_id`.

We fine-tune our models using pre-trained COCO models from [CenterNet](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md). Our HourglassNet model is fine-tuned from the pretrained [COCO CenterNet-HG model](https://drive.google.com/open?id=1cNyDmyorOduMRsgXoUnuyUiF6tZNFxaG). Our DLA model is fine-tuned from the pretrained [COCO CenterNet-DLA model](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT)
