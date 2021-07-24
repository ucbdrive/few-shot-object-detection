# Few-Shot Object Detection (FsDet) - Training tools for custom data

This is a fork of [Few-shot Object Detection (FsDet)](https://github.com/ucbdrive/few-shot-object-detection), adding an easy to use tool for training on custom datasets. The original FsDet readme with installation instructions can be found [here](README_fsdet.md).

## Concept and Features

We have extended the FsDet framework with a tool that dynamically generates datasets from annotation files and drives the training process. 

The tool has the following features:

- Determine the base and novel classes from the provided annotations (for the novel classes only a subset may be used for training).
- Determine how many instances are available, and set up the k-shot n-way problem accordingly.
- Prepare model structures for novel only and combined base+novel finetuning by adjusting the layer sizes to match the number of classes in the different sets. 
- If the number of samples strongly varies, set up multiple training problems to make best use of the data, and run multiple fine-tuning steps.

The tool currently supports annotations in COCO format. However, this does not mean that COCO is required as a base model, as long as the annotations are provided in this format. 

## Files

```data/custom_dataset.py``` implements a class holding all required configuration for the training task, based on a configuration file. The implements all necessary function to provide the dataset specific information to ```builtin.py```, ```builtin_meta.py```, ```coco_evaluation.py``` and ```ckpt_surgery.py```.

```dataset/prepare_custom_few_shot.py``` implements the dataset preparation and generation of configuration files. The generated configuration files are written to ```configs/custom_datasets

```train_few_shot.py``` is the driver of the process and the entry point to invoke the tool.

## Usage and configuration

```train_few_shot.py``` takes two arguments:
- ```--datasetconfig```: path to the configuration file for the training problem
- ```--ignoreunknown```: ignore annotations referring to classes not listed in the categories list of the annotation file
- ```--splitfact```: split into two few shot training problems with different k, if the 50% of classes with more samples have > splitfact x minimum number of samples

The configuration file is a YAML file with the following entries:
- base: the base dataset
-- classes_subset: a list class IDs of the categories found in the trainval file of the base dataset, which have been trained into the provided model.
-- model: the trained model file for the base classes
-- test: COCO format annotation file for the test data
-- test_dir: image based directory for the test data
-- trainval:  COCO format annotation file for the training/validation data
-- trainval_dir: image based directory for the training/validation data
- idoffset: integer to add to IDs in the novel categories to avoid clashes in numbering
- name: name for the combined dataset (will be used in configuration file and model names)
- maxk: maximum number of samples per class to use (mostly useful if testing few-shot training with data from a large dataset), if omitted or set to -1, the number of classes will be determined from the available data
- novel: the novel dataset
-- classes_subset: a list class IDs of the categories found in the trainval file of the novel dataset, which shall be used in the few shot training
-- test: COCO format annotation file for the test data
-- test_dir: image based directory for the test data
-- trainval:  COCO format annotation file for the training/validation data
-- trainval_dir: image based directory for the training/validation data

## Incremental training

For incrementatally adding classes, the tool produces three output files that can be used as new inputs for the next training step:

- ```datasets/<datasetname>/annotations/trainval-merged.json``` The combined training and validation data of the base and novel data
- ```datasets/<datasetname>/annotations/test-merged.json``` The combined test data of the base and novel data
- ```configs/custom_datasets/<configfilename>_base.yaml``` A configuration file containing the description of the new base dataset and classes

## Example

The example performs few shot training by adding a subset of 20 novel classes selected from the LVIS dataset to a pretrained based model containing 60 COCO classes (the base model can be downloaded from the [FsDet Model Zoo](https://github.com/ucbdrive/few-shot-object-detection/blob/master/docs/MODEL_ZOO.md)).

The example configuration file is located at ```configs/custom_datasets/coco_lvis1.yaml```.

The training is started with ```python train_few_shot.py --datasetconfig configs/custom_datasets/coco_lvis1.yaml --ignoreunknown```.


### Acknowledgement

This work has received funding from the European Union’s Horizon 2020 Research and Innovation Programme under grant agreement No 951911 ([AI4Media](https://www.ai4media.eu/)).
