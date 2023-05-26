#!/usr/bin/env python3
"""FSOD Torchserve Inference Handler



"""
import os
import time
import numpy as np
import torch
from types import SimpleNamespace
import logging

from ts.torch_handler.object_detector import ObjectDetector

from torchvision import transforms


from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

from fsdet.config import get_cfg
from fsdet.config import CfgNode
from fsdet.data import custom_dataset
# import to register default datasets
from fsdet.data import builtin
from fsdet.engine import DefaultPredictor

import yaml
from pathlib import Path
from PIL import Image
import io


from detectron2.data.detection_utils import read_image, convert_PIL_to_numpy


class FSObjectDetector(ObjectDetector):

    image_processing = transforms.Compose([
        transforms.ToTensor()
    ])
    
   

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()

        
        # NOTE: make sure the file does not contain single quotes - use double quotes instead
        loaded_cfg_dict = CfgNode.load_yaml_with_base( args.config_file, allow_unsafe=True )
        try:
            
            
            loaded_cfg = CfgNode(loaded_cfg_dict, new_allowed=True)

            
        except Exception as e:
            print("Exception when converting to CfgNode")
            print(e)
        try:
            cfg.merge_from_other_cfg(loaded_cfg)
        except Exception as e:
            print("Exception when merging CfgNodes")
            print(e)
                
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.freeze()
        
        
        return cfg

    def initialize(self, context):
    
        properties = context.system_properties
        
        
        self.map_location = "cuda" if torch.cuda.is_available(
        ) and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")
        

        conf = yaml.safe_load(Path(self.manifest["model"]["modelName"]+'_config.yml').read_text())

        
        self.threshold = conf["threshold"]


        self.args = SimpleNamespace()
        self.args.confidence_threshold = self.threshold
        self.args.config_file=conf["fsdet_config"]
        self.args.custom_dataset=None
        if "custom_dataset" in conf.keys():
            self.args.custom_dataset = conf["custom_dataset"]


        if not (self.args.custom_dataset == None):
            custom_dataset.register_all_custom(self.args.custom_dataset,"datasets")  
         
        self.cfg = self.setup_cfg(self.args)

        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")

        self.parallel = False

        
        self.predictor = DefaultPredictor(self.cfg)

        self.initialized = True


    def preprocess(self, data):

        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                print("service assumes one image as input")
                image = None

            images.append(image)

        return images
        
        
    def inference(self, data, *args, **kwargs):
        
        img = data[0]
        if img.is_cuda:
            img = img.cpu()
            
        topil = transforms.ToPILImage()
        img = topil(img)
        
        #img = read_image('000000000001.jpg', format="BGR")
        
        img = convert_PIL_to_numpy(img, format="BGR")
 
        predictions = self.predictor(img)

        
        return predictions
        
    def postprocess(self,data):
        result = []
               
        instances = data['instances']

        
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

        labels = metadata.thing_classes
        
        confident_detections = instances[instances.scores > self.threshold]

        
        for k in range(len(instances)):
            instk = instances[[k]]
        
            retval = {}
            clabel = labels[instk.pred_classes[0]]
            
            box = instk.pred_boxes[0].tensor
            
            if box.is_cuda:
                box = box.cpu()
                
            score = instk.scores[0]
            if score.is_cuda:
                score = score.cpu()
            score = score.item()
            
            retval[clabel] = box.tolist()
            retval['score'] = score

            result.append(retval)
            
        return [result]
                       

    


   
