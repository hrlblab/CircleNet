# Getting Started

This document provides tutorials to train and evaluate CircleNet. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).

## Benchmark evaluation

First, download the models you want to evaluate from our [model zoo](MODEL_ZOO.md) and put them in `CircleNet_ROOT/models/`. 

### MoNuSeg 2018

#### HG
To evaluate MoNuSeg object detection with HG, run

~~~
python main.py circledet exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4 --dataset monuseg --load_model ../models/circledet_monuseg_hg.pth --test --ontestdata --debug 4
~~~

This will give an AP of `48.7` if setup correctly. 

To evaluate the rotation consistency, run 

~~~
python main.py circledet --exp_id CircleNet_HG_Reproduce --arch hourglass --batch_size 4 --master_batch 4 --lr 2.5e-4   --dataset monuseg --load_model ../models/circledet_monuseg_hg.pth --test --ontestdata --rotate_reproduce 90
~~~

This will give '0.8918255489424661' if setup properly. The qualitative results can be found in 'exp/circledet/CircleNet_HG_Reproduce/debug'

#### DLA

To test with DLA, run

~~~
python main.py circledet --exp_id CircleNet_DLA_Reproduce --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4  --dataset monuseg --load_model ../models/circledet_monuseg_dla.pth --test --ontestdata --debug 4
~~~

This will give an AP of '48.6' if set-up properly. 

Rotation consistency can be evaluated by appending '--rotate_reproduce 90'. This should give '0.8855951478392722'.

## Training

All the training scripts can be found in [experiments](../experiments) folder.
The experiment names correspond to the model name in the [model zoo](MODEL_ZOO.md).

By default, pytorch evenly splits the total batch size to each GPUs.
`--master_batch` allows using different batchsize for the master GPU, which usually costs more memory than other GPUs.

If the training is terminated before finishing, you can use the same commond with `--resume` to resume training. It will found the latest model with the same `exp_id`.

We fine-tune our models using pre-trained COCO models from [CenterNet](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md). Our HourglassNet model is fine-tuned from the pretrained [COCO CenterNet-HG model](https://drive.google.com/open?id=1cNyDmyorOduMRsgXoUnuyUiF6tZNFxaG). Our DLA model is fine-tuned from the pretrained [COCO CenterNet-DLA model](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT)
