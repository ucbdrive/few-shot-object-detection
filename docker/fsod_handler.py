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
from fsdet.engine import DefaultPredictor

import yaml
from pathlib import Path


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
            
            loaded_cfg_dict = {}
            
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

        print("CUSTOM DS ")
        print(self.args.custom_dataset)


        if not (self.args.custom_dataset == None):
            custom_dataset.register_all_custom(self.args.custom_dataset,"datasets")

        print("DS TEST 0")
        print(cfg.DATASETS.TEST)
        
        print("-------------")
       
       
        self.cfg = self.setup_cfg(self.args)

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")

        self.parallel = False
        
        print(self.cfg)
        
        self.predictor = DefaultPredictor(self.cfg)

        self.initialized = True

        
        
    def inference(self, data, *args, **kwargs):

        predictions = self.predictor(image)
        
        # TODO - get predictions into right format for TS Detector output
        print(predictions)
        
        return predictions
                       
        
    


   
