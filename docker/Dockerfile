FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
ENV PYTHONUNBUFFERED TRUE

RUN apt-get update && apt-get -y upgrade && apt-get -y install git && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential checkinstall
RUN DEBIAN_FRONTEND=noninteractive apt-get autoremove
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev ca-certificates 
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install openjdk-11-jre-headless python3.8-venv python3-venv 
WORKDIR /opt
RUN wget https://www.python.org/ftp/python/3.8.6/Python-3.8.6.tgz
RUN tar xzf Python-3.8.6.tgz
RUN rm -f Python-3.8.6.tgz
WORKDIR /opt/Python-3.8.6
RUN ./configure --enable-optimizations
RUN make altinstall
RUN python3.8 -V

RUN apt-get -y install python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /

RUN python3 -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# setup tools version that avoids distutil version issue
# see https://github.com/pytorch/pytorch/issues/69894
RUN pip install setuptools==59.5.0
RUN pip install build

# rename to avoid clashes with other packages
RUN mv /usr/local/lib/python3.8/dist-packages/tools /usr/local/lib/python3.8/dist-packages/dt2_tools

# fsdet build and install wheel
RUN mkdir /workspace && \
  cd /workspace && \ 
  git clone https://github.com/wbailer/few-shot-object-detection.git && \
  cd few-shot-object-detection && \
  python3 -m pip install -r requirements.txt && \
  python -m build && \
  python -m pip install ./dist/fsdet-0.1-py3-none-any.whl 

WORKDIR /

# torchserve
RUN python -m pip install --no-cache-dir captum torchtext torchserve torch-model-archiver nvgpu pyyaml pathlib
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

RUN python -m pip install termcolor numpy tqdm matplotlib termcolor yacs tabulate cloudpickle Pillow imagesize tensorboard opencv-python lvis 

RUN mkdir -p /home/model-server/tmp
    
COPY config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store 

# copy model store
COPY model_store/* /home/model-server/model-store/

EXPOSE 8080 8081 8082 7070 7071

ENV TEMP=/home/model-server/tmp

# FSDET server

RUN python3 -m pip install flask flask-CORS

COPY fsserver.py /workspace/few-shot-object-detection/fsserver.py
COPY dockerd-entrypoint.sh /dockerd-entrypoint.sh
RUN chmod +x /workspace/few-shot-object-detection/fsserver.py
RUN chmod +x /dockerd-entrypoint.sh

ENV FSDET_ROOT=/workspace/few-shot-object-detection/
ENV MODEL_STORE=/home/model-server/model-store/

RUN apt-get update && apt-get -y install npm

RUN cd /workspace && \ 
  git clone https://github.com/wbailer/make-sense.git && \
  cd make-sense && \
  npm install
  
# copy required metadata, configs and models
RUN mkdir /workspace/few-shot-object-detection/datasets/cocosplit
COPY data_stage/cocosplit/ /workspace/few-shot-object-detection/datasets/cocosplit

COPY data_stage/configs/ /workspace/few-shot-object-detection/
COPY data_stage/models/ /workspace/few-shot-object-detection/

# copy model store
COPY model_store/* /home/model-server/model-store/

# prepare directory to be mounted
RUN mkdir /workspace/few-shot-object-detection/datasets/coco

WORKDIR /

CMD /bin/sh
#CMD ./dockerd-entrypoint.sh
