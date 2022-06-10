# InterMOT

## Introduction

Multiple object tracking (MOT) is a significant task in achieving autonomous driving. Previous fusion methods usually fuse the top-level features after the backbones extract the features from different modalities. In this paper, we first introduce PointNet++ to obtain multi-scale deep representations of point cloud to make it adaptive to our proposed Interactive Feature Fusion between multi-scale features of images and point clouds. Specifically, through multi-scale interactive query and fusion between pixel-level and point-level features, our method, can obtain more distinguishing features to improve the performance of multiple object tracking. 

For more details, please refer our [paper](https://arxiv.org/abs/2203.16268).

## Install

This project is based on pytorch==1.1.0, you can install it following the [official guide](https://pytorch.org/get-started/locally/).

We recommand you to build a new conda environment to run the projects as follows:
```bash
conda create -n intermot python=3.7 cython
conda activate intermot
conda install pytorch torchvision -c pytorch
conda install numba
```

Then install packages from pip:
```bash
pip install -r requirements.txt
```


## Usage

To train the model on your own, you can run command
```bash
python -u main.py --config ${work_path}/config.yaml --result-path=${work_path}/results 
```
## Data

We provide the data split used in our paper in the `data` directory. You need to download and unzip the data from the [KITTI Tracking Benchmark](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and put them in the `kitti_t_o` directory or any path you like.
Do remember to change the path in the configs.

## Acknowledgement

This code benefits a lot from [SECOND](https://github.com/traveller59/second.pytorch) and use the detection results provided by [MOTBeyondPixels](https://github.com/JunaidCS032/MOTBeyondPixels). The GHM loss implementation is from [GHM_Detection](https://github.com/libuyu/GHM_Detection).
