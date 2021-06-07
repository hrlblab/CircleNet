# CircleNet: Anchor-free Detection with Circle Representation
The official implementation of CircleNet, MICCAI 2020
### [[PyTorch]](https://github.com/hrlblab/CircleNet) [[project page]](https://github.com/hrlblab/CircleNet)  [[MICCAI paper]](https://arxiv.org/pdf/2006.02474.pdf)

Object detection networks are powerful in computer vision, but not
necessarily optimized for biomedical object detection. In this work, we propose
CircleNet, a simple anchor-free detection method with circle representation for
detection of the ball-shaped glomerulus. Different from the traditional bounding
box based detection method, the bounding circle (1) reduces the degrees of freedom of detection representation, (2) is naturally rotation invariant, (3) and optimized for ball-shaped objects. 

full citation is

Haichun Yang, Ruining Deng, Yuzhe Lu, Zheyu Zhu, Ye Chen, Joseph T. Roland, Le Lu, Bennett A. Landman, Agnes B. Fogo, and Yuankai Huo. "CircleNet: Anchor-free Detection with Circle Representation." arXiv preprint arXiv:2006.02474 (2020).

### Envrioment Set up
We used PyTorch 0.4.1. 

The implementation is based on the [CenterNet](https://github.com/xingyizhou/CenterNet) project.


Please follow the instructions adapted from [here](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md) to set up the environment.


1. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name CircleNet python=3.6
    ~~~
    
    And activate the environment.
    
    ~~~
    conda activate CircleNet
    ~~~

2. Install pytorch0.4.1:

    ~~~
    conda install pytorch=0.4.1 cuda92 torchvision -c pytorch
    ~~~
    
    And disable cudnn batch normalization(Due to [this issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).
    
    ~~~
    # PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
    # for pytorch v0.4.0
    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    # for pytorch v0.4.1
    sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    ~~~
     
    For other pytorch version, you can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. We observed slight worse training results without doing so. 
     
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

4. Clone this repo:

    ~~~
    CircleNet_ROOT=/path/to/clone/CircleNet
    git clone https://github.com/hrlblab/CircleNet.git $CircleNet_ROOT
    ~~~

5. Install the requirements
    
    ~~~
    pip install -r requirements.txt
    ~~~ 
    
6. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).

    ~~~
    cd $CircleNet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~
   
4. Compile NMS.

    ```
    cd $CircleNet_ROOT/src/lib/external
    make
    ```


### Testing on a whole slide image
The Case 03-1.scn file is avilable
https://vanderbilt.box.com/s/s530m45rvk626xi1thwcdc2bhoea758r

The model_10.pth model file is available (human kidney)
https://vumc.box.com/s/wpar2kz9600h9ao3wowjzc3y50znneop

To run it on a testing scan, please go to "src" folder and run
```
python run_detection_for_scn.py circledet --arch dla_34 --demo "/media/huoy1/48EAE4F7EAE4E264/Projects/from_haichun/batch_1_data/scn/Case 03-1.scn" --load_model /media/huoy1/48EAE4F7EAE4E264/Projects/detection/CircleNet/exp/circledet/kidpath_dla_batch4/model_10.pth --filter_boarder --demo_dir "/media/huoy1/48EAE4F7EAE4E264/Projects/detection/test_demo"
```

The demo_dir is output dir, which you set anywhere in your computer.

After running code, you will see a Case 03-1.xml file.
Then you put the xml and scn files into the same folder, and open the scn file using ImageScope software (only avilable in Windows OS), you can see something like the following image, with green detection results.

<img src="https://github.com/yuankaihuo/temp/blob/master/screenshot.jpg" width="60%" /> 

### A Google Colab demo of the above testing code is added 
https://github.com/hrlblab/CircleNet/blob/master/src/circle_net_demo.ipynb

### Run your own training code
The training code is
```
python main.py circledet --exp_id kidpath_dla_batch4 --arch dla_34 --batch_size 4 --master_batch 4 --lr 2.5e-4   --gpus 0 --print_iter 1  --dataset kidpath --save_all --load_model ../models/ctdet_coco_dla_2x.pth
```

You can get the ctdet_coco_dla_2x.pth model from model zoo
https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md
