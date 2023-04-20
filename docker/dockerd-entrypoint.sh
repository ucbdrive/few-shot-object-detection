#!/bin/bash

# rerun installation of fsdet wheel
pip install /workspace/few-shot-object-detection/dist/fsdet-0.1-py3-none-any.whl 

cd /workspace/few-shot-object-detection
python3 fsserver.py &


torchserve --ncs --start --ts-config /home/model-server/config.properties --models fsod_base_coco80=fsod_base_coco80.mar #&

#cd /workspace/make-sense
#npm start 


