# custom dataset using COCO annotations

import io
import yaml
import json
import numpy as np
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

import contextlib
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

class CustomDataset:

    def __init__(self):
   
        self.datasetinfo = {}
	   
        # init   
        self.datasetinfo['name'] = ''
        self.datasetinfo['idoffset'] = 100
        self.datasetinfo['maxk'] = -1
        self.datasetinfo['base'] = {}
        self.datasetinfo['base']['trainval'] = ''
        self.datasetinfo['base']['test'] = ''
        self.datasetinfo['base']['trainval_dir'] = ''
        self.datasetinfo['base']['test_dir'] = ''
        self.datasetinfo['base']['classes'] = {}
        self.datasetinfo['base']['classes_subset'] = [  ]
        self.datasetinfo['base']['classes_novel'] = [] 
        self.datasetinfo['base']['model'] = ''

	   
        self.datasetinfo['novel'] = {}
        self.datasetinfo['novel']['classes'] = {}
        self.datasetinfo['novel']['classes_subset'] = [  ]
        self.datasetinfo['novel']['trainval'] = ''
        self.datasetinfo['novel']['test'] = ''
        self.datasetinfo['novel']['trainval_dir'] = ''
        self.datasetinfo['novel']['test_dir'] = ''

	   
    def parse_classes(self):
        anno_base = json.load(open(self.datasetinfo['base']['trainval']))
	   
        self.datasetinfo['base']['classes'] = {}
        for c in anno_base['categories']:
            if len(self.datasetinfo['base']['classes_subset'])>0:
                if c['id'] in self.datasetinfo['base']['classes_subset']:
                    self.datasetinfo['base']['classes'][c['id']] = c['name']
            else:
                self.datasetinfo['base']['classes'][c['id']] = c['name']

        anno_novel = json.load(open(self.datasetinfo['novel']['trainval']))
	   
        self.datasetinfo['novel']['classes'] = {}
        self.datasetinfo['novel']['classcounts'] = {}
        for c in anno_novel['categories']:

            if len(self.datasetinfo['novel']['classes_subset'])>0:
                if c['id'] in self.datasetinfo['novel']['classes_subset']:
                    self.datasetinfo['novel']['classes'][c['id']] = c['name']
                    if 'instance_count' in c:
                        self.datasetinfo['novel']['classcounts'][c['id']] = c['image_count']
                    else:
                        self.datasetinfo['novel']['classcounts'][c['id']] = -1
    
            else:
                if 'instance_count' in c:
                    self.datasetinfo['novel']['classcounts'][c['id']] = c['image_count']
                else:
                    self.datasetinfo['novel']['classcounts'][c['id']] = -1
                    
                    
            # count annotations if data not provided
            if c['id'] in self.datasetinfo['novel']['classcounts'].keys():
                if self.datasetinfo['novel']['classcounts'][c['id']] == -1:
           
                    self.datasetinfo['novel']['classcounts'][c['id']] = 0
           
                    img_ids = {}
                    for a in anno_novel['annotations']:
                        if a['category_id'] == c['id']:
                            img_ids[a['image_id']] = 1
                            
                    self.datasetinfo['novel']['classcounts'][c['id']] = len(img_ids.keys())


       
  
    def serialise(self, filename):
   
        with io.open(filename, 'w', encoding='utf8') as outfile:
            yaml.dump(self.datasetinfo, outfile, default_flow_style=False, allow_unicode=True)
			
    def parse(self, filename, skipAnnofiles=False):

        with open(filename, 'r') as stream:
            self.datasetinfo = yaml.safe_load(stream)
            
        if not ('maxk' in self.datasetinfo.keys()):
            self.datasetinfo['maxk'] = -1
			
        if not skipAnnofiles:
            self.parse_classes()

    def get_adjusted_novel_classes(self):
        novelclasses = dict(self.datasetinfo['novel']['classes'])
        adjnovelclasses = {}
        for c in novelclasses.keys():
            adjnovelclasses[c+self.get_id_offset()] = novelclasses[c]
            
        return adjnovelclasses

    def get_id2class(self):
        allclasses = dict(self.datasetinfo['base']['classes'])
        novelclasses = self.get_adjusted_novel_classes()
        allclasses.update(novelclasses)
        return allclasses
	
    def get_base_class_ids(self): 
        return list(self.datasetinfo['base']['classes'].keys())

    def get_novel_class_ids(self,adjusted=True):
        if adjusted: 
            novelclasses = self.get_adjusted_novel_classes()
            return list(novelclasses.keys())
        else: 
            return self.datasetinfo['novel']['classes']
        
    def get_base_train_annotation_file(self):
        return self.datasetinfo['base']['trainval']
        
    def get_base_test_annotation_file(self):
        return self.datasetinfo['base']['test']

    def get_base_train_dir(self):
        return self.datasetinfo['base']['trainval_dir']
        
    def get_base_test_dir(self):
        return self.datasetinfo['base']['test_dir']    
		
    def get_novel_train_annotation_file(self):
        return self.datasetinfo['novel']['trainval']
        
    def get_novel_test_annotation_file(self):
        return self.datasetinfo['novel']['test']

    def get_novel_train_dir(self):
        return self.datasetinfo['novel']['trainval_dir']
        
    def get_novel_test_dir(self):
        return self.datasetinfo['novel']['test_dir']  
        
    def get_id_offset(self):
        return self.datasetinfo['idoffset']
        
    def get_name(self):
        return self.datasetinfo['name']
    
    def get_nshots(self):
        if self.datasetinfo['maxk']>0:
            return min(min(self.datasetinfo['novel']['classcounts'].values()),self.datasetinfo['maxk'])
        else:
            return min(self.datasetinfo['novel']['classcounts'].values())
        
    def get_nclasses_all(self):
        return len(self.datasetinfo['base']['classes']) + self.get_nclasses_novel()
    
    def get_nclasses_novel(self):
        return len(self.datasetinfo['novel']['classes'])
    
    def get_base_model_file(self):
        return self.datasetinfo['base']['model']
    
    def get_config_file(self,cfgtype):
        if not(cfgtype=='all' or cfgtype=='novel'):
            print('unknown configuration file type: '+cfgtype)
   
   
        if cfgtype == 'novel':
            modelsuffix = 'remove'
        else:
            modelsuffix = 'combine'
        
        weightname = '"models/fs/faster_rcnn_R_101_FPN_' + self.get_name() + "/model_reset_" + modelsuffix + '.pth"'
        
        cfgstr1 = \
        """_BASE_: "../Base-RCNN-FPN.yaml"
MODEL:
  WEIGHTS: """
        
        
        cfgstr1b = \
"""
  MASK_ON: False
  RESNETS:
    DEPTH: 101
  ROI_HEADS:
    NUM_CLASSES: """
    
        cfgstr1_all = \
"""    OUTPUT_LAYER: "CosineSimOutputLayers"
"""
        cfgstr2 = \
"""
    FREEZE_FEAT: True
  BACKBONE:
    FREEZE: True
  PROPOSAL_GENERATOR:
    FREEZE: True
"""

        cfgstr_data_all = "DATASETS:\n  TRAIN: ('" + self.get_name() + '_trainval_all_' + str(self.get_nshots()) + 'shot' + "',)\n  TEST: ('" +self.get_name() + '_test_all' + "',)\n"

        cfgstr_data_novel = "DATASETS:\n  TRAIN: ('" + self.get_name() + '_trainval_novel_' + str(self.get_nshots()) + 'shot' + "',)\n  TEST: ('" +self.get_name() + '_test_novel' + "',)\n"

  
        cfgstr_solver_all = \
"""
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.001
  STEPS: (14400,)
  MAX_ITER: 16000
  CHECKPOINT_PERIOD: 1000
  WARMUP_ITERS: 10
OUTPUT_DIR: "models/fs/faster_rcnn_R_101_FPN_"""     
        cfgstr_solver_novel = \
"""
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.01
  STEPS: (10000,)
  MAX_ITER: 500
  CHECKPOINT_PERIOD: 500
  WARMUP_ITERS: 0      
OUTPUT_DIR: "models/fs/faster_rcnn_R_101_FPN_"""       
    
        if cfgtype=='all':
            cfgstr = cfgstr1 + weightname + cfgstr1b + str(self.get_nclasses_all()) + '\n' + cfgstr1_all + cfgstr2  + cfgstr_data_all + cfgstr_solver_all + self.get_name() + '"\n'
        if cfgtype=='novel':
            cfgstr = cfgstr1 + weightname + cfgstr1b + str(self.get_nclasses_novel()) + '\n' + cfgstr2 + cfgstr_data_novel + cfgstr_solver_novel + self.get_name() + '/novel"\n'
        
    
        return cfgstr
        
        
    def get_base_categories_color(self):
        return self.get_categories_color('base')
    
    def get_novel_categories_color(self):
        return self.get_categories_color('novel')
    
    def get_categories_color(self,setname):
        clsdict = self.datasetinfo[setname]['classes']
        if setname == 'novel':
            clsdict = self.get_adjusted_novel_classes()
        
        colorcatlist = []
        
        clist = self.get_random_colors(len(clsdict))
        
        i = 0
        for clsid in clsdict.keys():
            catdict = {}
            catdict['color'] = clist[i]
            catdict['isthing'] = 1
            catdict['id'] = clsid
            catdict['name'] = clsdict[clsid]
            colorcatlist.append(catdict)

            i = i+1

        return colorcatlist
        
    def get_random_colors(self,ncol):
        colorvec = np.random.random(size=3*ncol) * 216 + 40
        colormat = np.reshape(colorvec,(ncol,3))
        return colormat.tolist()
        
        
    def create_merged_base_model(self):
        cdm = CustomDataset()
        
        cdm.datasetinfo['name'] = self.datasetinfo['name'] + '_merged'
        cdm.datasetinfo['idoffset'] = self.datasetinfo['idoffset'] + self.get_nclasses_all()
        cdm.datasetinfo['base'] = {}
        cdm.datasetinfo['base']['trainval'] = os.path.join('datasets',self.get_name(),'annotations','trainval-merge.json')
        cdm.datasetinfo['base']['test'] = os.path.join('datasets',self.get_name(),'annotations','trainval-merge.json')
        cdm.datasetinfo['base']['trainval_dir'] = '.'
        cdm.datasetinfo['base']['test_dir'] = '.'
        cdm.datasetinfo['base']['classes'] = {}
        cdm.datasetinfo['base']['classes_subset'] = self.datasetinfo['base']['classes_subset'] + self.datasetinfo['novel']['classes_subset']
        cdm.datasetinfo['base']['model'] = os.path.join('models','fs','faster_rcnn_R_101_FPN_'+self.get_name(),'model_final.pth')
        
        return cdm
        
# util functions

# for builtin and custom meta

def register_all_custom(cfgfilename,root="datasets"):
    #cfgfilename = os.path.join('configs','custom_datasets','coco_lvis1.yaml')

    cds = CustomDataset()

    cds.parse(cfgfilename)
    
    
    # for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO_SN.items():
    #     for key, (image_root, json_file) in splits_per_dataset.items():
    #         # Assume pre-defined datasets live in `./datasets`.
    #         register_coco_instances(
    #             key,
    #             _get_custom_builtin_metadata(dataset_name,cds),
    #             os.path.join(root, json_file)
    #             if "://" not in json_file
    #             else json_file,
    #             os.path.join(root, image_root),
    #         )




    # register meta datasets
    METASPLITS = [
        (
            cds.get_name() + "_trainval_all",
            "",
            cds.get_name() + "/annotations/trainval-merged.json",
        ),
        (
            cds.get_name() + "_trainval_base",
            "",
            cds.get_name() + "/annotations/trainval-merged.json",
        ),
        (cds.get_name() + "_test_all", "", cds.get_name() + "/annotations/test-merged.json"),
        (cds.get_name() + "_test_base", "", cds.get_name() + "/annotations/test-merged.json"),
        (cds.get_name() + "_test_novel", "", cds.get_name() + "/annotations/test-merged.json"),
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for shot in [ cds.get_nshots() ]:
            for seed in range(10):
                seed = "" if seed == 0 else "_seed{}".format(seed)
                name = cds.get_name() + "_trainval_{}_{}shot{}".format(prefix, shot, seed)
                METASPLITS.append((name, "", ""))


    for name, imgdir, annofile in METASPLITS:
        register_meta_custom(
            name,
            _get_custom_builtin_metadata(cds.get_name() + "_fewshot",cds),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
            cds.get_name()
        )
        

def load_custom_json(json_file, image_root, metadata, dataset_name, dsname):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    is_shots = "shot" in dataset_name
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", dsname)
        if "seed" in dataset_name:
            shot = dataset_name.split("_")[-2].split("shot")[0]
            seed = int(dataset_name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = dataset_name.split("_")[-1].split("shot")[0]
        for idx, cls in enumerate(metadata["thing_classes"]):
            json_file = os.path.join(
                split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls)
            )
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
    else:
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))
    id_map = metadata["thing_dataset_id_to_contiguous_id"]
    
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]

    if is_shots:
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    
                    record["file_name"] = os.path.join(
                        image_root, img_dict["file_name"]
                    )
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    
                    # ignore categories not in set
                    if obj["category_id"] not in id_map.keys():
                        continue
                    
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)
    else:
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts


def register_meta_custom(name, metadata, imgdir, annofile, dsname):

    try:
        DatasetCatalog.get(name)
        # if dataset is found, return
        return
    except KeyError as e:
        found = False

    DatasetCatalog.register(
        name,
        lambda: load_custom_json(annofile, imgdir, metadata, name, dsname ),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]
        
        print("thing len of "+name)
        print(metadata["thing_classes"])

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/"+dsname,
        **metadata,
    )
    
# for builtin meta


def _get_custom_builtin_metadata(dataset_name,cds):
    if dataset_name == cds.get_name():
        return _get_custom_instances_meta(cds)
    elif dataset_name == cds.get_name() + "_fewshot":
        return _get_custom_fewshot_instances_meta(cds)

    raise KeyError("No metadata for dataset {}".format(dataset_name))
		
def _get_custom_instances_meta(cds):
    CUSTOM_CATEGORIES = cds.get_base_categories_color() + cds.get_novel_categories_color()

    thing_ids = [k["id"] for k in CUSTOM_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in CUSTOM_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == cds.get_nclasses_all(), len(thing_ids)
    # Mapping from the incontiguous id to continous id
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CUSTOM_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_custom_fewshot_instances_meta(cds):
    CUSTOM_NOVEL_CATEGORIES = cds.get_novel_categories_color()
    CUSTOM_CATEGORIES = cds.get_base_categories_color()

    ret = _get_custom_instances_meta(cds)
    novel_ids = [k["id"] for k in CUSTOM_NOVEL_CATEGORIES if k["isthing"] == 1]
    novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}
    novel_classes = [
        k["name"] for k in CUSTOM_NOVEL_CATEGORIES if k["isthing"] == 1
    ]
    base_categories = [
        k
        for k in CUSTOM_CATEGORIES
        if k["isthing"] == 1 and k["name"] not in novel_classes
    ]
    base_ids = [k["id"] for k in base_categories]
    base_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(base_ids)}
    base_classes = [k["name"] for k in base_categories]
    ret[
        "novel_dataset_id_to_contiguous_id"
    ] = novel_dataset_id_to_contiguous_id
    ret["novel_classes"] = novel_classes
    ret["base_dataset_id_to_contiguous_id"] = base_dataset_id_to_contiguous_id
    ret["base_classes"] = base_classes
    return ret
		
