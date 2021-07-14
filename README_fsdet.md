# Few-Shot Object Detection (FsDet)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ucbdrive/few-shot-object-detection.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ucbdrive/few-shot-object-detection/context:python)

FsDet contains the official few-shot object detection implementation of the ICML 2020 paper
[Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/abs/2003.06957).
![TFA Figure](https://user-images.githubusercontent.com/7898443/76520006-698cc200-6438-11ea-864f-fd30b3d50cea.png)

In addition to the benchmarks used by previous works, we introduce new benchmarks on three datasets: PASCAL VOC, COCO, and LVIS. We sample multiple groups of few-shot training examples for multiple runs of the experiments and report evaluation results on both the base classes and the novel classes. These are described in more detail in [Data Preparation](#data-preparation).

We also provide benchmark results and pre-trained models for our two-stage fine-tuning approach (TFA). In TFA, we first train the entire object detector on the data-abundant base classes, and then only fine-tune the last layers of the detector on a small balanced training set. See [Models](#models) for our provided models and [Getting Started](#getting-started) for instructions on training and evaluation.

FsDet is well-modularized so you can easily add your own datasets and models. The goal of this repository is to provide a general framework for few-shot object detection that can be used for future research.

If you find this repository useful for your publications, please consider citing our paper.

```angular2html
@article{wang2020few,
    title={Frustratingly Simple Few-Shot Object Detection},
    author={Wang, Xin and Huang, Thomas E. and  Darrell, Trevor and Gonzalez, Joseph E and Yu, Fisher}
    booktitle = {International Conference on Machine Learning (ICML)},
    month = {July},
    year = {2020}
}
```

## Updates
The code has been upgraded to detectron2 v0.2.1.  If you need the original released code, please checkout the release [v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags) in the tag.  



## Table of Contents
- [Installation](#installation)
- [Code Structure](#code-structure)
- [Data Preparation](#data-preparation)
- [Models](#models)
- [Getting Started](#getting-started)


## Installation

**Requirements**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.4
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* CUDA 10.0, 10.1, 10.2
* GCC >= 4.9

**Build FsDet**
* Create a virtual environment.
```angular2html
python3 -m venv fsdet
source fsdet/bin/activate
```
You can also use `conda` to create a new environment.
```angular2html
conda create --name fsdet
conda activate fsdet
```
* Install Pytorch 1.6 with CUDA 10.2 
```angular2html
pip install torch torchvision
```
You can choose the Pytorch and CUDA version according to your machine.
Just to make sure your Pytorch version matches the [prebuilt detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#install-pre-built-detectron2-linux-only)
* Install Detectron2 v0.2.1
```angular2html
python3 -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html
```
* Install other requirements. 
```angular2html
python3 -m pip install -r requirements.txt
```

## Code Structure
- **configs**: Configuration files
- **datasets**: Dataset files (see [Data Preparation](#data-preparation) for more details)
- **fsdet**
  - **checkpoint**: Checkpoint code.
  - **config**: Configuration code and default configurations.
  - **engine**: Contains training and evaluation loops and hooks.
  - **layers**: Implementations of different layers used in models.
  - **modeling**: Code for models, including backbones, proposal networks, and prediction heads.
- **tools**
  - **train_net.py**: Training script.
  - **test_net.py**: Testing script.
  - **ckpt_surgery.py**: Surgery on checkpoints.
  - **run_experiments.py**: Running experiments across many seeds.
  - **aggregate_seeds.py**: Aggregating results from many seeds.


## Data Preparation
We evaluate our models on three datasets:
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/): We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits. The splits can be found in [fsdet/data/datasets/builtin_meta.py](fsdet/data/datasets/builtin_meta.py).
- [COCO](http://cocodataset.org/): We use COCO 2014 and extract 5k images from the val set for evaluation and use the rest for training. We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.
- [LVIS](https://www.lvisdataset.org/): We treat the frequent and common classes as the base classes and the rare categories as the novel classes.

See [datasets/README.md](datasets/README.md) for more details.


## Models
We provide a set of benchmark results and pre-trained models available for download in [MODEL_ZOO.md](docs/MODEL_ZOO.md).


## Getting Started

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](fsdet/model_zoo/model_zoo.py),
  for example, `COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml`.
2. We provide `demo.py` that is able to run builtin standard models. Run it with:
```
python3 -m demo.demo --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS fsdet://coco/tfa_cos_1shot/model_final.pth
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.

### Training & Evaluation in Command Line

To train a model, run
```angular2html
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml
```

To evaluate the trained models, run
```angular2html
python3 -m tools.test_net --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
        --eval-only
```

For more detailed instructions on the training procedure of TFA, see [TRAIN_INST.md](docs/TRAIN_INST.md).

### Multiple Runs

For ease of training and evaluation over multiple runs, we provided several helpful scripts in `tools/`.

You can use `tools/run_experiments.py` to do the training and evaluation. For example, to experiment on 30 seeds of the first split of PascalVOC on all shots, run
```angular2html
python3 -m tools.run_experiments --num-gpus 8 \
        --shots 1 2 3 5 10 --seeds 0 30 --split 1
```

After training and evaluation, you can use `tools/aggregate_seeds.py` to aggregate the results over all the seeds to obtain one set of numbers. To aggregate the 3-shot results of the above command, run
```angular2html
python3 -m tools.aggregate_seeds --shots 3 --seeds 30 --split 1 \
        --print --plot
```
