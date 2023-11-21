from fsdetdatasets.prepare_custom_few_shot import main as preparefct
from fsdettools.train_net import _main as trainfct
from fsdettools.ckpt_surgery import main as surgeryfct

import os
import sys
import shutil
import glob
import copy

from fsdet.data.custom_dataset import CustomDataset

    
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
    
    print('\nCOMPLETED')


def train_fs(args):


    cdsconfig = args.datasetconfig
    ignoreunknown = args.ignoreunknown
    
    cds = CustomDataset()
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

   
    ### Ensemble entry point here ###
    if args.ensemble:
        print('preparing ensembled classifiers')
        surgeryfct(['ckpt_surgery',
                    '--src1', 'checkpoints/coco/base_model/model_final.pth',
                    '--method','ensemble',
                    '--ensemble_config', 'configs/COCO-detection/'+cds.get_name()+'.yaml'],
                   # '--ensemble_config', 'configs/COCO-detection/test.yaml'])
                    '--num-heads', args.num_heads)

        print('fine-tuning')
        trainfct(['train_net',
                  '--config-file', 'configs/COCO-detection/'+cds.get_name()+'.yaml',
                  # '--config-file', 'configs/COCO-detection/test.yaml',
                  '--num-heads', args.num_heads,
                  '--opts', 'MODEL.WEIGHTS',
                  'checkpoints/coco/base_model/model_reset_ensemble.pth'])

        print('\nCOMPLETED')