from datasets.prepare_custom_few_shot import main as preparefct
from tools.train_net import _main as trainfct
from tools.ckpt_surgery import main as surgeryfct

import argparse
import os
import sys
import shutil
import glob
import copy

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../fsdet/data'))
sys.path.append(lib_path)

import custom_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetconfig", type=str, help="YAML file specifying the dataset")
    parser.add_argument("--ignoreunknown", action='store_true', default=False, help="ignore classes in the annotations not listed in categories  ")
    parser.add_argument("--splitfact", type=float, default=-1, help="split the training problem, if 50% of the classes have fact more samples than the minimum. -1 means never split")
    args = parser.parse_args()
    return args
    
    
def process_training_task(cds,cdsconfig,ignoreunknown):


    print('prepare dataset')
    
    prepargs = ['prepare_custom_few_shot','--datasetconfig',cdsconfig]
    if ignoreunknown:
        prepargs.append('--ignoreunknown')

    preparefct(prepargs)
    
    
    seedid = 1
    srcfiles = glob.glob(os.path.join('datasets',cds.get_name(),'seed'+str(seedid),'*.json'))
    for f in srcfiles:    
        shutil.copy2(f,os.path.join('datasets',cds.get_name(),''))

        
        
    print('prepare model')
    surgeryfct(['ckpt_surgery','--src1',cds.get_base_model_file(),'--method','remove','--custom',cdsconfig,'--save-dir','models/fs/faster_rcnn_R_101_FPN_'+cds.get_name()])        
        
    print('few shot training')
       
    trainfct(['train_net','--config-file','configs/custom_datasets/faster_rcnn_R_101_FPN_ft_novel_fshot_'+cds.get_name()+'.yaml','--custom_datacfg','configs/custom_datasets/'+cds.get_name()+'.yaml','--opts','MODEL.WEIGHTS','models/fs/faster_rcnn_R_101_FPN_'+cds.get_name()+'/model_reset_remove.pth'])
 
   
    print('combine model')

    surgeryfct(['ckpt_surgery','--src1',cds.get_base_model_file(),'--src2','models/fs/faster_rcnn_R_101_FPN_'+cds.get_name()+'/novel/model_final.pth','--method','combine','--custom',cdsconfig,'--save-dir','models/fs/faster_rcnn_R_101_FPN_'+cds.get_name()])

    print('fine-tuning')
    
    trainfct(['train_net','--config-file','configs/custom_datasets/faster_rcnn_R_101_FPN_ft_all_fshot_'+cds.get_name()+'.yaml','--custom_datacfg','configs/custom_datasets/'+cds.get_name()+'.yaml','--opts','MODEL.WEIGHTS','models/fs/faster_rcnn_R_101_FPN_'+cds.get_name()+'/model_reset_combine.pth'])


def main(arglist):

    sys.argv = arglist
    sys.argc = len(arglist)

    args = parse_args()

    cdsconfig = args.datasetconfig
    ignoreunknown = args.ignoreunknown
    
    cds = custom_dataset.CustomDataset()
    cds.parse(cdsconfig)
    
    # decide about splitting training task
    if args.splitfact > 0:
         minK = cds.get_nshots()
         fracLarger = 0
         fewclasses = []
         moreclasses = []
         
         for c in cds.datasetinfo['novel']['classcounts'].keys():
             if cds.datasetinfo['novel']['classcounts'][c] > minK*args.splitfact:
                 fracLarger = fracLarger +1
                 moreclasses.append(c)
             else:
                 fewclasses.append(c)
         
         fracLarger = fracLarger / cds.get_nclasses_novel()
         if fracLarger > 0.5:
             print("splitting the task")
         
             cds1 = copy.deepcopy(cds)
             cds1.datasetinfo['name'] = cds.get_name() + '_A'
             cds1.datasetinfo['novel']['classes_subset'] = moreclasses
             cds1name = os.path.join('configs','custom_datasets',cds1.get_name()+'.yaml')
             cds1.serialise(cds1name)
      
             
             cds2 = copy.deepcopy(cds)
             cds2.datasetinfo['name'] = cds.get_name() + '_B'
             cds2.datasetinfo['novel']['classes_subset'] = fewclasses
             cds2.datasetinfo['base']['trainval'] = os.path.join('datasets',cds.get_name(),'annotations','trainval-merge.json')
             cds2.datasetinfo['base']['test'] = os.path.join('datasets',cds.get_name(),'annotations','test-merge.json')
             cds2.datasetinfo['base']['trainval_dir'] = '.'
             cds2.datasetinfo['base']['test_dir'] = '.'
             cds2.datasetinfo['base']['model'] = os.path.join('models','fs','faster_rcnn_R_101_FPN_'+cds1.get_name(),'model_final.pth')
             cds2name = os.path.join('configs','custom_datasets',cds2.get_name()+'.yaml')
             cds2.serialise(cds2name)

             process_training_task(cds1,cds1name,ignoreunknown)   
             process_training_task(cds2,cds2name,ignoreunknown)   
             
         else: 
             process_training_task(cds,cdsconfig,ignoreunknown)   
                  
    else:
        process_training_task(cds,cdsconfig,ignoreunknown)

if __name__ == '__main__':
    main(sys.argv)
   
