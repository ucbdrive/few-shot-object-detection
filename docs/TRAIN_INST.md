# TFA Training Instructions

TFA is trained in two stages. We first train the entire object detector on the data-abundant base classes, and then only fine-tune the last layers of the detector on a small balanced training set. We provide detailed instructions for each stage.

![TFA Figure](https://user-images.githubusercontent.com/7898443/76520006-698cc200-6438-11ea-864f-fd30b3d50cea.png)

## Stage 1: Base Training

First train a base model. To train a base model on the first split of PASCAL VOC, run
```angular2html
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_base1.yaml
```

<details>
<summary>COCO</summary>

```angular2html
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_base.yaml
```

</details>

## Stage 2: Few-Shot Fine-Tuning

### Initialization

After training the base model, run ```tools/ckpt_surgery.py``` to obtain an initialization for the full model. We only modify the weights of the last layer of the detector, while the rest of the network are kept the same. The weights corresponding to the base classes are set as those obtained in the previous stage, and the weights corresponding to the novel classes are either randomly initialized or set as those of a predictor fine-tuned on the novel set.

#### Random Weights

To randomly initialize the weights corresponding to the novel classes, run
```angular2html
python3 -m tools.ckpt_surgery \
        --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method randinit \
        --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1
```
The resulting weights will be saved to `checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1/model_reset_surgery.pth`.

<details>
<summary>COCO</summary>

```angular2html
python3 -m tools.ckpt_surgery \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth \
        --method randinit \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all
        --coco
```

</details>

#### Novel Weights

To use novel weights, fine-tune a predictor on the novel set. We reuse the base model trained in the previous stage but retrain the last layer from scratch. On PASCAL VOC, we found novel weights to be unnecessary so we did not use them and do not provide the config files. You can see [here](https://github.com/ucbdrive/few-shot-object-detection/issues/13#issuecomment-614865673) for an example if you would still like to use it. Below we provide instructions on the COCO dataset.

First remove the last layer from the weights file by running
```angular2html
python3 -m tools.ckpt_surgery \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth \
        --method remove \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all
```

Next, fine-tune the predictor on the novel set by running
```angular2html
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_novel_1shot.yaml \
        --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_remove.pth
```

Finally, combine the base weights from the base model with the novel weights by running
```angular2html
python3 -m tools.ckpt_surgery \
        --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth \
        --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/model_final.pth \
        --method combine \
        --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all
        --coco
```
The resulting weights will be saved to `checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_combine.pth`.

<details>
<summary>PASCAL VOC</summary>

Just for reference, not actually used.
```angular2html
python3 -m tools.ckpt_surgery \
        --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method remove \
        --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1
```

```angular2html
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_novel1_1shot.yaml \
        --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1/model_reset_remove.pth
```

```angular2html
python3 -m tools.ckpt_surgery \
        --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --src2 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel1_1shot/model_final.pth \
        --method combine \
        --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1
```

</details>

### Fine-Tuning

We will then fine-tune the last layer of the full model on a balanced dataset by running
```angular2html
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
        --opts MODEL.WEIGHTS $WEIGHTS_PATH
```
where `WEIGHTS_PATH` is the path to the weights obtained in the previous initialization step.

<details>
<summary>COCO</summary>

```angular2html
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
        --opts MODEL.WEIGHTS $WEIGHTS_PATH
```

</details>

## Training Steps for Each Dataset

Below are the steps we used during training for each dataset.
```
PASCAL VOC: Base training --> random initialization --> fine-tuning
COCO and LVIS: Base training --> novel weights initializaiton --> fine-tuning
```
