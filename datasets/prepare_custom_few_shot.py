# preparation script for custom dataset

import argparse
import json
import os
import random
import gc
import sys
from urllib.parse import urlparse

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../fsdet/data'))
sys.path.append(lib_path)

import custom_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 10],
                        help="Range of seeds")
    parser.add_argument("--datasetconfig", type=str, help="YAML file specifying the dataset")
    parser.add_argument("--ignoreunknown", action='store_true', default=False, help="ignore classes in the annotations not listed in categories  ")
    args = parser.parse_args()
    return args


def generate_seeds(args,cds,ignoreUnknown,ID2CLASS,CLASS2ID):
    data_path_base = cds.get_base_train_annotation_file()
    data_path_novel = cds.get_novel_train_annotation_file()
    
    img_dir_base = cds.get_base_train_dir()
    img_dir_novel = cds.get_novel_train_dir()

    
    data_base = json.load(open(data_path_base))
    data_novel = json.load(open(data_path_novel))

    # merge the JSON structures
    data = data_base
    
    for c in data_novel['categories']:
        c['id'] = c['id'] + cds.get_id_offset()
    
    data['categories'] = data['categories'] + data_novel['categories']
    
    for i in data['images']:
        # patch for mixed COCO train/val
        if 'COCO' in i['file_name']:
            valpath = img_dir_base.replace('train','val')
            if 'val' in i['file_name']:
                i['file_name'] = os.path.join(valpath,i['file_name'])
            else:
                i['file_name'] = os.path.join(img_dir_base,i['file_name'])
        else:
            i['file_name'] = os.path.join(img_dir_base,i['file_name'])

    for i in data_novel['images']:
        # get filename from COCO URL if not explicitly present
        fn = ''
        mypath = img_dir_novel
        if not('file_name' in i.keys()):
            purl = urlparse(i['coco_url'])
            fn = os.path.basename(purl.path)
            if 'val' in i['coco_url']:
                mypath = img_dir_novel.replace('train','val')
        else:
            fn = i['file_name']

        i['file_name'] = os.path.join(mypath,fn)
    
    data['images'] = data['images'] + data_novel['images']

    for a in data_novel['annotations']:
        a['category_id'] = a['category_id'] + cds.get_id_offset()


    data['annotations'] = data['annotations'] + data_novel['annotations']

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    
    for a in data['annotations']:
        if not(a['category_id'] in anno.keys()):
            continue
            
        if 'iscrowd' in a.keys():
            if a['iscrowd'] == 1:
                continue
        anno[a['category_id']].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for c in ID2CLASS.keys():
            print('class '+str(c))
            img_ids = {}
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]

            sample_shots = []
            sample_imgs = []
            
            #print("class "+str(c)+", nshots "+str(cds.get_nshots())+ " nimgs "+str(len(img_ids)))

            
            for shots in [ cds.get_nshots() ]:
   
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        #if len(img_ids[img]) + len(sample_shots) > shots:
                        #    continue
                        # adjust for missing number of annotation
                        if len(img_ids[img]) > shots - len(sample_shots):
                            sample_shots.extend(img_ids[img][0:shots - len(sample_shots)])
                            print(str(len(sample_shots)))
                        else:
                            sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                new_data = {
                    'info': data['info'],
                    'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(ID2CLASS[c], shots, i, cds)
                new_data['categories'] = new_all_cats
                print('saving new data to '+save_path)
                with open(save_path, 'w') as f:
                    json.dump(new_data, f)

def generate_merged_test(cds):

    data_path_base = cds.get_base_test_annotation_file()
    data_path_novel = cds.get_novel_test_annotation_file()
    
    base_ids = cds.get_base_class_ids()
    novel_ids = cds.get_novel_class_ids(False)
    
    img_dir_base = cds.get_base_test_dir()
    img_dir_novel = cds.get_novel_test_dir()

    
    data_base_test = json.load(open(data_path_base))

    # merge the JSON structures
    datamerged = data_base_test

    for i in datamerged['images']:
        i['file_name'] = os.path.join(img_dir_base,i['file_name'])

    filtered_cats = []

    for c in datamerged['categories']:
        if c['id'] in base_ids:
            filtered_cats.append(c)
    datamerged['categories'] = filtered_cats
    
    # novel val
    data_novel_test = json.load(open(data_path_novel))

    
    filtered_cats = []

    for c in data_novel_test['categories']:
        if c['id'] in novel_ids:
            filtered_cats.append(c)
            c['id'] = c['id'] + cds.get_id_offset()
    data_novel_test['categories'] = filtered_cats
    

    datamerged['categories'] = datamerged['categories'] + data_novel_test['categories']

    for i in data_novel_test['images']:
        # get filename from COCO URL if not explicitly present
        fn = ''
        mypath = img_dir_novel
        if not('file_name' in i.keys()):
            purl = urlparse(i['coco_url'])
            fn = os.path.basename(purl.path)
            if 'train' in i['coco_url']:
                mypath = img_dir_novel.replace('val','train')
        else:
            fn = i['file_name']

        i['file_name'] = os.path.join(mypath,fn)
        
    datamerged['images'] = datamerged['images'] + data_novel_test['images']
    
    filtered_annots = []

    for a in datamerged['annotations']:
        if a['category_id'] in base_ids:
            filtered_annots.append(a)
    datamerged['annotations'] = filtered_annots
    
    filtered_annots = []

    for a in data_novel_test['annotations']:
        if a['category_id'] in novel_ids:
            filtered_annots.append(a)
            a['category_id'] = a['category_id'] + cds.get_id_offset()
    data_novel_test['annotations'] = filtered_annots
    
    datamerged['annotations'] = datamerged['annotations'] + data_novel_test['annotations']
 
    for a in datamerged['annotations']:
         if not('iscrowd' in a.keys()):
            a['iscrowd'] = 0
 
    
    del data_novel_test
    gc.collect()


    outdir = os.path.join('datasets',cds.get_name(),'annotations')

    if not(os.path.exists(outdir)):
        os.makedirs(outdir)
    with open(os.path.join(outdir,'test-merged.json'),'w') as jsonfile:
        json.dump(datamerged,jsonfile)
       
        
    del datamerged
    gc.collect()
    

def generate_merged_trainval(cds):

    data_path_base = cds.get_base_train_annotation_file()
    data_path_novel = cds.get_novel_train_annotation_file()
    
    base_ids = cds.get_base_class_ids()
    novel_ids = cds.get_novel_class_ids(False)

    
    img_dir_base = cds.get_base_train_dir()
    img_dir_novel = cds.get_novel_train_dir()

    
    data_base_trainval= json.load(open(data_path_base))

    # merge the JSON structures
    datamerged = data_base_trainval

    for i in datamerged['images']:
        # patch for mixed COCO train/val
        if 'COCO' in i['file_name']:
            valpath = img_dir_base.replace('train','val')
            if 'val' in i['file_name']:
                i['file_name'] = os.path.join(valpath,i['file_name'])
            else:
                i['file_name'] = os.path.join(img_dir_base,i['file_name'])
        else:
            i['file_name'] = os.path.join(img_dir_base,i['file_name'])

    filtered_cats = []

    for c in datamerged['categories']:
        if c['id'] in base_ids:
            filtered_cats.append(c)
    datamerged['categories'] = filtered_cats

    
    # novel val
    data_novel_trainval = json.load(open(data_path_novel))

    
    filtered_cats = []

    for c in data_novel_trainval['categories']:
        if c['id'] in novel_ids:
            filtered_cats.append(c)
            c['id'] = c['id'] + cds.get_id_offset()
    data_novel_trainval['categories'] = filtered_cats

    datamerged['categories'] = datamerged['categories'] + data_novel_trainval['categories']

    for i in data_novel_trainval['images']:
        # get filename from COCO URL if not explicitly present
        fn = ''
        mypath = img_dir_novel
        if not('file_name' in i.keys()):
            purl = urlparse(i['coco_url'])
            fn = os.path.basename(purl.path)
            if 'val' in i['coco_url']:
                mypath = img_dir_novel.replace('train','val')
        else:
            fn = i['file_name']

        i['file_name'] = os.path.join(mypath,fn)
        
    datamerged['images'] = datamerged['images'] + data_novel_trainval['images']
    
    filtered_annots = []

    for a in datamerged['annotations']:
        if a['category_id'] in base_ids:
            filtered_annots.append(a)
    datamerged['annotations'] = filtered_annots
    
    filtered_annots = []

    for a in data_novel_trainval['annotations']:
        if a['category_id'] in novel_ids:
            filtered_annots.append(a)
            a['category_id'] = a['category_id'] + cds.get_id_offset()
    data_novel_trainval['annotations'] = filtered_annots

    datamerged['annotations'] = datamerged['annotations'] + data_novel_trainval['annotations']
 
    
    del data_novel_trainval
    gc.collect()


    outdir = os.path.join('datasets',cds.get_name(),'annotations')

    if not(os.path.exists(outdir)):
        os.makedirs(outdir)
    with open(os.path.join(outdir,'trainval-merged.json'),'w') as jsonfile:
        json.dump(datamerged,jsonfile)
       
        
    del datamerged
    gc.collect()

def get_save_path_seeds( cls, shots, seed, cds):
    #s = path.split('/')
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, cls)
    save_dir = os.path.join('datasets', cds.get_name(), 'seed' + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path

def main(arglist):

    sys.argv = arglist
    sys.argc = len(arglist)

    args = parse_args()

    cds = custom_dataset.CustomDataset()

    cds.parse(args.datasetconfig)
	
    #cds.serialise(args.datasetconfig)
	
    ID2CLASS = cds.get_id2class()
    
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    generate_seeds(args,cds,args.ignoreunknown,ID2CLASS,CLASS2ID)
    
    generate_merged_test(cds)
    
    generate_merged_trainval(cds)
    
    # write config files
    cfgfilename = 'faster_rcnn_R_101_FPN_ft_novel_fshot_' + cds.get_name() + '.yaml'

    with open(os.path.join('configs','custom_datasets',cfgfilename), 'w') as f:
        f.write(cds.get_config_file('novel'))
        
    cfgfilename = 'faster_rcnn_R_101_FPN_ft_all_fshot_' + cds.get_name() + '.yaml'
    with open(os.path.join('configs','custom_datasets',cfgfilename), 'w') as f:
        f.write(cds.get_config_file('all'))
    
    
    modeldir = os.path.join('models','fs','faster_rcnn_R_101_FPN_'+cds.get_name())
    
    if not(os.path.exists(modeldir)):
        os.makedirs(modeldir)
     
    # updated training file for incremental training
    cdm = cds.create_merged_base_model()
    cdm.serialise(os.path.join('configs','custom_datasets',cds.get_name()+'_merged.yaml'))
        

if __name__ == '__main__':
    main(sys.argv)

