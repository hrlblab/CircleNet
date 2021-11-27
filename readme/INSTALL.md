## Installation

We used PyTorch 0.4.1 on Ubuntu 18.04 LTS with [Anaconda](https://www.anaconda.com/download) Python 3.6.

The implementation and instructions are based on the [CenterNet](https://github.com/xingyizhou/CenterNet) project.

1. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name CircleNet python=3.6
    ~~~
    
    And activate the environment.
    
    ~~~
    conda activate CircleNet
    ~~~

2. Install Pytorch 0.4.1:

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
     
3. Install the requirements
    ~~~
    pip install -r requirements.txt
    ~~~ 

4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

5. Clone this repo:

    ~~~
    CircleNet_ROOT=/path/to/clone/CircleNet
    git clone https://github.com/hrlblab/CircleNet.git $CircleNet_ROOT
    ~~~


    
6. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).

    ~~~
    cd $CircleNet_ROOT/src/lib/models/networks/DCNv2git 
    ./make.sh
    ~~~
   
7. Compile NMS.

    ```
    cd $CircleNet_ROOT/src/lib/external
    make
    ```