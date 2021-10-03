# Custom Dataset Instructions

Here we provide instructions for how to create and use your own custom dataset for training and evaluation.

## Step 1: Create dataset loader

The first step is to create your dataset loader in the `fsdet/data` directory. The dataset loader should return a list of all the images in the dataset and their corresponding annotations. The format is shown below:
```python
def dataset_loader(name, thing_classes):
    data = []
    for file in dataset:
        annotation = [
            "file_name" : "x.png", # full path to image
            "image_id" :  0, # image unique ID
            "height" : 123, # height of image
            "width" : 123, # width of image
            "annotations": [
                "category_id" : thing_classes.index("class_name"), # class unique ID
                "bbox" : [0, 0, 123, 123], # bbox coordinates
                "bbox_mode" : BoxMode.XYXY_ABS, # bbox mode, depending on your format
            ]
        ]
        data.append(annotation)
    return data
```
For the bbox format, you see refer to the Detectron2 [code](https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/boxes.py#L23) for the available options.

For examples, you can refer to the PASCAL VOC [dataset loader](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/data/meta_pascal_voc.py#L12) or the COCO [dataset loader](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/data/meta_coco.py#L19).

## Step 2: Create meta information for dataset

Next, you have to create the meta information for your dataset. The only needed information is the classes for each split. Below is an example of the classes information for the first split of PASCAL VOC:
```python
thing_classes: [ # all classes
    "aeroplane", "bicycle", "boat", "bottle", "car", "cat", "chair",
    "diningtable", "dog", "horse", "person", "pottedplant", "sheep",
    "train", "tvmonitor", "bird", "bus", "cow", "motorbike", "sofa",
]
base_classes: [ # base clases
    "aeroplane", "bicycle", "boat", "bottle", "car", "cat", "chair",
    "diningtable", "dog", "horse", "person", "pottedplant", "sheep",
    "train", "tvmonitor",
]
novel_classes: ["bird", "bus", "cow", "motorbike", "sofa"] # novel classes

metadata = {
    "thing_clases": thing_clases,
    "base_classes": base_classes,
    "novel_classes": novel_classes,
}
```

We put all the meta information in [builtin_meta.py](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/data/builtin_meta.py).

## Step 3: Create evaluator for dataset

Now you need to write a evaluator to evaluate on your dataset. Your class should inherit from `DatasetEvaluator` and implement all its functions. Example below:

```python
from fsdet.evaluation.evaluator import DatasetEvaluator
class NewDatasetEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name): # initial needed variables
        self._dataset_name = dataset_name

    def reset(self): # reset predictions
        self._predictions = []

    def process(self, inputs, outputs): # prepare predictions for evaluation
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                prediction["instances"] = output["instances"]
            self._predictions.append(prediction)

    def evaluate(self): # evaluate predictions
        results = evaluate_predictions(self._predictions)
        return {
            "AP": results["AP"],
            "AP50": results["AP50"],
            "AP75": results["AP75"],
        }
```

For examples, you can refer to the PASCAL VOC [evaluator](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/evaluation/pascal_voc_evaluation.py) or the COCO [evaluator](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/evaluation/coco_evaluation.py).

## Step 4: Register dataset loader and meta information

For the rest of the code to see your new dataset, you need to register it with Detectron2's DatasetCatalog and MetadataCatalog. Example below:
```python
def register_dataset(name, thing_classes, metadata):
    # register dataset (step 1)
    DatasetCatalog.register(
        name, # name of dataset, this will be used in the config file
        lambda: dataset_loader( # this calls your dataset loader to get the data
            name, thing_classes, # inputs to your dataset loader
        ),
    )

    # register meta information (step 2)
    MetadataCatalog.get(name).set(
        thing_classes=metadata["thing_classes"], # all classes
        base_classes=metadata["base_classes"], # base classes
        novel_classes=metadata["novel_classes"], # novel classes
    )
    MetadataCatalog.get(name).evaluator_type = "new_dataset" # set evaluator
```

We put the above code in a register function in its corresponding dataset file. For examples, you can refer to the PASCAL VOC register [code](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/data/meta_pascal_voc.py#L135) or the COCO register [code](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/data/meta_coco.py#L124).

Then, you also need to call the register function to register the dataset. We do this in [builtin.py](https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/data/builtin.py). You should register all your datasets, including all base and novel splits. Example below:
```python
datasets = {
    'dataset_all': metadata["thing_classes"],
    'dataset_base': metadata["base_classes"],
    'dataset_novel': metadata["novel_classes"],
}
for dataset_name, classes in datasets.items():
    register_dataset(dataset_name, classes, metadata)
```

Also, add your evaluator to both `tools/train_net.py` and `tools/test_net.py` in the `build_evaluator` function:
```python
def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    ...
    if evaluator_type == "new_dataset":
        return NewDatasetEvaluator(dataset_name)
    ...
```

## Step 5: Modify config files

Modify the config files to use your new dataset. You only need to replace the datasets part to include the name of your new dataset. Example below:
```yaml
_BASE_: "../../Base-RCNN-FPN.yaml"
MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
  MASK_ON: False
  RESNETS:
    DEPTH: 101
  ROI_HEADS:
    NUM_CLASSES: 15
INPUT:
  MIN_SIZE_TRAIN: (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
  MIN_SIZE_TEST: 800
DATASETS:
  TRAIN: ('dataset_all',) # <-- modify this
  TEST: ('dataset_novel',) # <-- modify this
SOLVER:
  STEPS: (12000, 16000)
  MAX_ITER: 18000
  WARMUP_ITERS: 100
OUTPUT_DIR: "checkpoints/new_dataset"
```

Congratulations, now you can start training and testing on your new dataset!
