#!/bin/bash


#docker run --rm -dit --gpus 0 -p:3000:3000 -v /home/baw/fsdet/few-shot-object-detection/datasets_fs/tm2:/workspace/few-shot-object-detection/datasets/tm2 -v /home/baw/fsdet/few-shot-object-detection/datasets/cocosplit:/workspace/few-shot-object-detection/datasets/cocosplit fsdet_ms

docker run --rm -it --gpus 0 -p:3000:3000 -p 3010:3010 -p 8080:8080 -p 8081:8081 fsdet_ms

#docker run --rm -it --gpus 0 -p:3000:3000 -p 3010:3010 -p 8080:8080 -p 8081:8081 fsdet_ms:demo
