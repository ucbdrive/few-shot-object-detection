from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

import logging
import os
from detectron2.config import global_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from .lvis_v0_5_categories import LVIS_CATEGORIES_NOVEL_IDS

"""
This file contains functions to parse LVIS-format annotations into dicts in the
"Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["register_meta_lvis"]


def load_filtered_lvis_json(
    json_file, image_root, metadata, dataset_name=None
):
    """
    Load a json file in LVIS's annotation format.
    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(
                json_file, timer.seconds()
            )
        )

    if dataset_name is not None and "train" in dataset_name:
        assert global_cfg.MODEL.ROI_HEADS.NUM_CLASSES == len(
            metadata["thing_classes"]
        ), "NUM_CLASSES should match number of categories: ALL=1230, NOVEL=454"

    # sort indices for reproducible results
    img_ids = sorted(list(lvis_api.imgs.keys()))
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(
        ann_ids
    ), "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))

    logger.info(
        "Loaded {} images in the LVIS format from {}".format(
            len(imgs_anns), json_file
        )
    )

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        file_name = img_dict["file_name"]
        if img_dict["file_name"].startswith("COCO"):
            file_name = file_name[-16:]
        record["file_name"] = os.path.join(image_root, file_name)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get(
            "not_exhaustive_category_ids", []
        )
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            if global_cfg.MODEL.ROI_HEADS.NUM_CLASSES == 454:
                # Novel classes only
                if anno["category_id"] - 1 not in LVIS_CATEGORIES_NOVEL_IDS:
                    continue
                obj["category_id"] = metadata["class_mapping"][
                    anno["category_id"] - 1
                ]
            else:
                # Convert 1-indexed to 0-indexed
                obj["category_id"] = anno["category_id"] - 1
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_meta_lvis(name, metadata, json_file, image_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection.
    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset.
            It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    DatasetCatalog.register(
        name,
        lambda: load_filtered_lvis_json(json_file, image_root, metadata, name),
    )

    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="lvis",
        **metadata,
    )