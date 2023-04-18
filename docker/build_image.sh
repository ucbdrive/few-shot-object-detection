#!/bin/bash

mkdir data_stage

cp -r ../configs ./data_stage

cp -r ../datasets/cocosplit ./data_stage/

mkdir ./data_stage/models
cp -r ../models/coco ./data_stage/models

# make copy of base configs for Torchserve
mkdir ./data_stage/configs_ts
cp ../configs/COCO-detection/faster_rcnn_R_101_FPN_base.yaml ./data_stage/configs_ts/faster_rcnn_R_101_FPN_base.yaml
cp ../configs/COCO-detection/faster_rcnn_R_101_FPN_base80.yaml ./data_stage/configs_ts/faster_rcnn_R_101_FPN_base80.yaml

# fix path
sed -i 's/_BASE_: "\.\.\//_BASE_: "/g' ./data_stage/configs_ts/faster_rcnn_R_101_FPN_base.yaml
sed -i 's/_BASE_: "\.\.\//_BASE_: "/g' ./data_stage/configs_ts/faster_rcnn_R_101_FPN_base80.yaml
# replace single with double quotes
sed -i s/\'/\"/g ./data_stage/configs_ts/faster_rcnn_R_101_FPN_base.yaml
sed -i s/\'/\"/g ./data_stage/configs_ts/faster_rcnn_R_101_FPN_base80.yaml

torch-model-archiver -f --model-name fsod_base_coco80 --handler fsod_handler.py --extra-files ./data_stage/configs/Base-RCNN-FPN.yaml,./data_stage/configs_ts/faster_rcnn_R_101_FPN_base80.yaml,fsod_base_coco80_config.yml --export-path model_store -v 1.0 --serialized-file ./data_stage/models/coco/faster_rcnn_R_101_FPN_base80/model_final.pth

torch-model-archiver -f --model-name fsod_base_coco60 --handler fsod_handler.py --extra-files ./data_stage/configs/Base-RCNN-FPN.yaml,./data_stage/configs_ts/faster_rcnn_R_101_FPN_base.yaml,fsod_base_coco60_config.yml --export-path model_store -v 1.0 --serialized-file ./data_stage/models/coco/faster_rcnn_R_101_FPN_base/model_final.pth

rm -r ./data_stage/config_ts

# for demo

mkdir data_stage/tm2
cp -r ../datasets_fs/tm2 ./data_stage/

docker build -t fsdet_ms .

docker build -f Dockerfile.demo -t fsdet_ms:demo .

rm -r data_stage
