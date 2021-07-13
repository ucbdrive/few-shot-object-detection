"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Here we only register the few-shot datasets and complete COCO, PascalVOC and 
LVIS have been handled by the builtin datasets in detectron2. 
"""

import os

from typing import List

from .builtin_meta import _get_builtin_metadata
from .meta_coco import register_meta_coco
from fsdet.utils import io

PROJ_ROOT = str(io.get_project_root())
# TODO: This has to be injected either as an environment variable or a parameter.
DATASET_ROOT = os.path.join(PROJ_ROOT, 'datasets/socket_plates')
os.chdir(PROJ_ROOT)

# ==== Predefined datasets and splits for COCO ==========
_PREDEFINED_SPLITS_COCO = {}

_PREDEFINED_SPLITS_COCO["custom"] = {
    "custom_2014_train_aop": (
        os.path.join(DATASET_ROOT, "train"),
        os.path.join(DATASET_ROOT, "annotations/train.json")
    ),
    "custom_2014_val_aop": (
        os.path.join(DATASET_ROOT, "val"),
        os.path.join(DATASET_ROOT, "annotations/val.json")
    )
}


def register_all_coco(root=DATASET_ROOT):
    # register meta datasets
    METASPLITS = [
        (
            "custom_train_all_aop",
            os.path.join(DATASET_ROOT, "train"),
            os.path.join(DATASET_ROOT, "annotations/train.json")
        ),
        (
            "custom_train_base_aop",
            os.path.join(DATASET_ROOT, "val"),
            os.path.join(DATASET_ROOT, "annotations/val.json")
        ),
        (
            "custom_test_all_aop",
            os.path.join(DATASET_ROOT, "val"),
            os.path.join(DATASET_ROOT, "annotations/val.json")
        ),
        (
            "custom_test_base_aop",
            os.path.join(DATASET_ROOT, "val"),
            os.path.join(DATASET_ROOT, "annotations/val.json")
        ),
        (
            "custom_test_novel_aop",
            os.path.join(DATASET_ROOT, "val"),
            os.path.join(DATASET_ROOT, "annotations/val.json")
        )
    ]

    # register small meta datasets for fine-tuning stage
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10]:
            for seed in range(5):
                seed = "" if seed == 0 else "_seed{}".format(seed)
                name = "custom_train_{}_{}shot{}".format(prefix, shot, seed)
                METASPLITS.append((name, DATASET_ROOT + "/train", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("custom_fewshot"),
            os.path.join(PROJ_ROOT, imgdir),
            os.path.join(PROJ_ROOT, annofile),
        )


# Register them all under "./datasets"
register_all_coco()
