"""
Generate annotations to perform semi-supervised training.

Supports also generation GT annotations.
"""

import numpy as np
import torch

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import default_argument_parser, default_setup

import detectron2.utils.comm as comm
import json
import logging
import os
import time
import contextlib
import io
import copy
import random
import demo.demo

from collections import OrderedDict
from detectron2.data import MetadataCatalog
from detectron2.engine import hooks, launch

from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


# get list of filenames used in the subset of the dataset for those classes
def get_subdataset_file(dataset,classes):

    split_dir = dataset.dirname + 'split'

    if "seed" in dataset_name:
        shot = dataset_name.split("_")[-2].split("shot")[0]
        seed = int(dataset_name.split("_seed")[-1])
        split_dir = os.path.join(split_dir, "seed{}".format(seed))
    else:
        shot = dataset_name.split("_")[-1].split("shot")[0]

    all_img_ids = []

    for cls in classes:
        json_file = os.path.join(
            split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls)
        )
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        img_ids = list(coco_api.imgs.keys())
        all_img_ids.extend(img_ids)
        
    all_img_ids = list(dict.fromkeys(all_img_ids))
        
    return all_img_ids

def get_additional_gt_annotations(dataset,imglist,classes,gt_annos,max_nr = None,min_nr = None):
     
    split_dir = dataset.dirname + 'split'

    if "seed" in dataset_name:
        shot = dataset_name.split("_")[-2].split("shot")[0]
        seed = int(dataset_name.split("_seed")[-1])
        split_dir = os.path.join(split_dir, "seed{}".format(seed))
    else:
        shot = dataset_name.split("_")[-1].split("shot")[0]   
     
    # split into classes and serialise
    for idx, cls in enumerate(ds.thing_classes):
       
        
        if cls not in classes:
            continue

    
    
        catlist = gt_annos.getCatIds([cls])

        print(catlist)

        
        imgIds = gt_annos.getImgIds(imglist,catlist)
        annIds = gt_annos.getAnnIds(imgIds,catlist)
        if len(imgIds)==0:
            annIds = []
        

        
        if min_nr is not None:
            delta = min_nr-len(annIds)
            if delta>0:
                allimgs = gt_annos.getImgIds()
                catimgIds = gt_annos.getImgIds(allimgs,catlist)
                imgIds = imgIds + catimgIds
            annIds = gt_annos.getAnnIds(imgIds,catlist)



        
        if max_nr is not None:
            if len(annIds)>max_nr:
                imgIds = random.choices(imgIds,k=max_nr)
                annIds = gt_annos.getAnnIds(imgIds,catlist)
                if len(annIds)>max_nr:
                    annIds = random.choices(annIds,k=max_nr)
               
    
    
        filtered_annos = {}
        filtered_annos['info'] = copy.copy(gt_annos.dataset['info'])
        filtered_annos['categories'] = copy.copy(gt_annos.dataset['categories'])
        filtered_annos['images'] = gt_annos.loadImgs(imgIds)
        
         
        for img in filtered_annos['images']:
            if 'val' in img['coco_url']:
                img['coco_url'] = img['coco_url'].replace('/train','/val')
                if not img['file_name'].startswith('../val2014/'):
                    img['file_name'] = '../val2014/'+img['file_name']
                

        filtered_annos['annotations'] = gt_annos.loadAnns(annIds)

      
        ffile_name = os.path.join(
                split_dir, "full_box_{}shot_{}_ss_trainval.json".format(shot, cls)
            )
     
        with open(ffile_name,'w') as outfile:
            json.dump(filtered_annos,outfile)
            
  

def generate_ss_gt(dataset,ss_gt_file,max_nr = None,min_nr = None):


    base_images = get_subdataset_file(dataset,dataset.base_classes)
    novel_images = get_subdataset_file(dataset,dataset.novel_classes)
        
    json_file = PathManager.get_local_path(ss_gt_file)
    with contextlib.redirect_stdout(io.StringIO()):
        gt_anno_set = COCO(json_file)
        
    get_additional_gt_annotations(dataset,base_images,dataset.novel_classes,gt_anno_set,max_nr,min_nr)
    get_additional_gt_annotations(dataset,novel_images,dataset.base_classes,gt_anno_set,max_nr,min_nr)
    
def get_additional_annotations(dataset,imglist,classes,gt_annos,max_nr = None,min_nr = None, classifier_path=None, modelcfg=None, conf=0.1, renumber=False):
     
    split_dir = dataset.dirname + 'split'

    if "seed" in dataset_name:
        shot = dataset_name.split("_")[-2].split("shot")[0]
        seed = int(dataset_name.split("_seed")[-1])
        split_dir = os.path.join(split_dir, "seed{}".format(seed))
    else:
        shot = dataset_name.split("_")[-1].split("shot")[0]   
     
    stats = {}
     
    mappedcatid = 0
     
    # split into classes and serialise
    for idx, cls in enumerate(ds.thing_classes):
             
        
        if cls not in classes:
            continue
            
        # workaround for missing annotations
        if cls=="person":
           mappedcatid+=1
           continue
           
        # get same images used for ground truth
    
        catlist = gt_annos.getCatIds([cls])

        if not(renumber):
            mappedcatid = idx
        
        imgIds = gt_annos.getImgIds(imglist,catlist)
        annIds = gt_annos.getAnnIds(imgIds,catlist)
        if len(imgIds)==0:
            annIds = []
        

        
        if min_nr is not None:
            delta = min_nr-len(annIds)
            if delta>0:
                allimgs = gt_annos.getImgIds()
                catimgIds = gt_annos.getImgIds(allimgs,catlist)
                imgIds = imgIds + catimgIds
            annIds = gt_annos.getAnnIds(imgIds,catlist)



        
        if max_nr is not None:
            if len(annIds)>max_nr:
                imgIds = random.choices(imgIds,k=max_nr)
                annIds = gt_annos.getAnnIds(imgIds,catlist)
                if len(annIds)>max_nr:
                    annIds = random.choices(annIds,k=max_nr)
       
        filtered_annos = {}
        filtered_annos['info'] = copy.copy(gt_annos.dataset['info'])
        filtered_annos['categories'] = copy.copy(gt_annos.dataset['categories'])                  
        filtered_annos['annotations'] = []
        filtered_annos['images'] = []
        
        newImgs = copy.copy(imgIds)
        # force all new images
        #newImgs = []
        
        
        imgIds = []
    
        print("class "+str(cls)+": generating annotations")
        
        catid = catlist[0]
        assert catid < 91 # for COCO

            
        chklist = []
        
        initialSet = True
        nAddlImgs = 0
    
        while len(filtered_annos['annotations'])<min_nr:
            newimgobjs = gt_annos.loadImgs(newImgs) 
        
            print("got "+str(len(filtered_annos['annotations']))+", fetching more")
         
         
            for img in newimgobjs:
                if 'val' in img['coco_url']:
                    img['coco_url'] = img['coco_url'].replace('/train','/val')
                    if not img['file_name'].startswith('../val2014/'):
                        img['file_name'] = '../val2014/'+img['file_name']           
         
            if classifier_path==None:
                classifier_path="models/fs/faster_rcnn_R_101_FPN_fs1/model_final.pth"
 
            if modelcfg==None:
                modelcfg="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_fshot_baw2.yaml"

         
            for img in newimgobjs:
                demo.main(["--config-file",modelcfg,#"configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_fshot_baw2.yaml",
                          "--input", "~/fsdet/few-shot-object-detection/datasets/coco/train2014/"+img['file_name'],
                          "--confidence-threshold",str(conf),
                          "--output","result.png",
                          "--json","temp.json","--opts","MODEL.WEIGHTS",
                          classifier_path,
                          #"models/fs/faster_rcnn_R_101_FPN_fs1/novel/model_final.pth",
                          #"models/fs/faster_rcnn_R_101_FPN_fs1/model_final.pth",
                          #"models/fs/faster_rcnn_R_101_FPN_fs1_ss/model_final.pth",
                          ])
                          
                        
                jsonfile = open("temp.json")
                annots = json.load(jsonfile)
                
                print("\n\n\n\nfound "+str(len(annots['annotations']))+" annos ")
                
                for a in annots['annotations']:
  
                    print("searching "+str(mappedcatid)+" found "+str(a['category_id'])+" so far found "+str(len(filtered_annos["annotations"])))
                    
                    # for COCO
                    assert mappedcatid<81
                    assert a['category_id']<81
                    
                    if int(a['category_id']) != mappedcatid:
                        continue
                    a['category_id'] = catid
                    a['image_id'] = img['id']
                    filtered_annos['annotations'].append(a)

            if initialSet:
                stats[cls] = (len(filtered_annos['annotations']),0,0)
                initialSet = False
                         
            imgIds.extend(newImgs)
            filtered_annos['images'].extend(newimgobjs)
            newImgs = []
            
            if len(filtered_annos['annotations'])<min_nr:
                allimgs = gt_annos.getImgIds()
                catimgIds = gt_annos.getImgIds(allimgs,catlist)
                newImgs = random.choices(catimgIds,k=100)  
                
                nAddlImgs = nAddlImgs + len(newImgs)  
                
                
            #print(len(filtered_annos['annotations']))
            #exit(0)
    
        if max_nr is not None:
            if len(filtered_annos['annotations'])>max_nr:
                filtered_annos['annotations'] = random.choices(filtered_annos['annotations'],k=max_nr)    

        stats[cls] = (stats[cls][0],len(filtered_annos['annotations']) - stats[cls][0],nAddlImgs)


      
        ffile_name = os.path.join(
                split_dir, "ss007_ftsep/full_box_{}shot_{}_ss_trainval.json".format(shot, cls)
            )
     
        with open(ffile_name,'w') as outfile:
            json.dump(filtered_annos,outfile)
            
        mappedcatid += 1
   
    with open('stats.txt','a') as outfile:
        outfile.write(json.dumps(stats))
    
def generate_annotations(dataset,ss_gt_file,max_nr = None,min_nr = None, single_classifier=True):


    base_images = get_subdataset_file(dataset,dataset.base_classes)
    novel_images = get_subdataset_file(dataset,dataset.novel_classes)
        
    json_file = PathManager.get_local_path(ss_gt_file)
    with contextlib.redirect_stdout(io.StringIO()):
        gt_anno_set = COCO(json_file)
        
    #"models/coco/faster_rcnn_R_101_FPN_base/model_final.pth"
    #"models/fs/faster_rcnn_R_101_FPN_fs1/novel/model_final.pth",
    #"models/fs/faster_rcnn_R_101_FPN_fs1/model_final.pth",
    #"models/fs/faster_rcnn_R_101_FPN_fs1_ss/model_final.pth",
        
    classifier_path_novel = "models/fs/faster_rcnn_R_101_FPN_fs1/model_final.pth"
    classifier_path_base = "models/fs/faster_rcnn_R_101_FPN_fs1/model_final.pth"
    
    modelcfg_novel = "configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_fshot_baw2.yaml"
    modelcfg_base = "configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_fshot_baw2.yaml"

    
    if not(single_classifier):
        classifier_path_novel = "models/fs/faster_rcnn_R_101_FPN_fs1/novel/model_final.pth"
        classifier_path_base = "models/coco/faster_rcnn_R_101_FPN_base/model_final.pth"
        
        modelcfg_novel = "configs/COCO-detection/faster_rcnn_R_101_FPN_ft_novel_fshot_baw2.yaml"
        modelcfg_base = "configs/COCO-detection/faster_rcnn_R_101_FPN_base.yaml"
      
    novelconf = 0.07
    baseconf = novelconf
    
    if not(single_classifier):
        baseconf = 0.3

        
    get_additional_annotations(dataset,base_images,dataset.novel_classes,gt_anno_set,max_nr,min_nr,classifier_path_novel,modelcfg_novel,novelconf,not(single_classifier))
    
    get_additional_annotations(dataset,novel_images,dataset.base_classes,gt_anno_set,max_nr,min_nr,classifier_path_base,modelcfg_base,baseconf,not(single_classifier))

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    cfg = setup(args)

    dataset_name = cfg['DATASETS']['TRAIN'][0]

    ds = MetadataCatalog.get(dataset_name)
    
    # single classifier: same for novel and base, when False, the base classes classifier will resort to the base model
    single_classifier = False
    
    random.seed(42)
    
    # check if ground truth is set
    if args.ss_gt:
        max_nr = None
        min_nr = None
        if args.ss_max:
            max_nr = args.ss_max
        if args.ss_min:
            min_nr = args.ss_min
            if min_nr<max_nr:
                min_nr = max_nr
                print("WARNING: set min_nr to max_nr")
        generate_ss_gt(ds,args.ss_gt,max_nr,min_nr)
    elif args.ss_auto:
        max_nr = None
        min_nr = None
        if args.ss_max:
            max_nr = args.ss_max
        if args.ss_min:
            min_nr = args.ss_min
            if min_nr<max_nr:
                min_nr = max_nr
                print("WARNING: set min_nr to max_nr")
        generate_annotations(ds,args.ss_auto,max_nr,min_nr,single_classifier)
              
    
 
