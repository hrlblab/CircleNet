# CircleNet-Installation 2023 by juming xiong

## choose your cuda version
Please refer this [Website](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) to see which cuda version you can choose.

## Operating System
[recommended] Ubuntu-20.04 or Ubuntu-18.04.

If you are using the Windows System, the best way to use Linux system is [Install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

## Install gcc and g++
update apt 
~~~
sudo apt update
~~~
install gcc and g++
~~~
sudo apt install gcc g++
~~~

## Anaconda/Miniconda 
Download Miniconda
~~~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
~~~
Install Miniconda
~~~
sh Miniconda3-latest-Linux-x86_64.sh
~~~
restart terminal session
~~~
source .bashrc
~~~

## Setup virtual environment
Recommend python version 3.7 or 3.6,  pytorch version > 1.7
for example:
~~~
conda create -n CircleNet python=3.7
~~~
activate the environment
~~~
conda activate CircleNet
~~~
install Pytorch 1.11 and cudatoolkit 11.3
~~~
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
~~~

## Install Packages
install Cython and pycocotools
~~~
pip install cython pycocotools
~~~
clone this repo
~~~
git clone https://github.com/hrlblab/CircleNet.git
~~~
install requirements
~~~
cd CircleNet
pip install -r requirements.txt
~~~
Install [COCOAPI](https://github.com/cocodataset/cocoapi):
~~~
COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git 
cd $COCOAPI/PythonAPI
make
python setup.py install --user
~~~

## Compile deformable convolutional (from [DCNv2](https://github.com/lbin/DCNv2/tree/pytorch_1.11))
~~~
cd $CircleNet_ROOT/src/lib/models/networks/
rm -rf DCNv2/
git clone -b <branch_name> https://github.com/lbin/DCNv2.git # for pytorch 1.11 is: git clone -b pytorch_1.11 https://github.com/lbin/DCNv2.git
cd DCNv2
./make.sh
~~~

## Compile NMS
~~~
cd $CircleNet_ROOT/src/lib/external
make
~~~

## [Optional] For Whole Image Slide demo
There are still some steps need to do
1. install some packages
   ~~~
   sudo apt update
   pip install openslide-python
   sudo apt install python3-openslide or python-openslide
   ~~~
   if failed to install python-openslide, please tyoe ```sudo nano /etc/apt/sources.list``` and add ```deb http://dk.archive.ubuntu.com/ubuntu/ bionic main universe``` to the file. Then use ```sudo apt update``` and ``` sudo apt install python-openslide```

2. if you meet the problem about PIL
   please install Pillow==6.2
   ~~~
   pip install Pillow==6.2
   ~~~

3. if you meet the problem about libffi.so.7
   please add ```export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7``` to your ```.bashrc``` file



