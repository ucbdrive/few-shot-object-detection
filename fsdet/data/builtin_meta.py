# All coco custom dataset categories
COCO_CATEGORIES = [

    {'supercategory': '', 'id': 1, 'name': 'AOP_EVK80'},
    {'supercategory': '', 'id': 2, 'name': 'AOP_TRAS1000'},
    {'supercategory': '', 'id': 3, 'name': 'AOP_TRAS1000_no_key'},
    {'supercategory': '', 'id': 5, 'name': 'SPLITTER_MCP_03'},
    {'supercategory': '', 'id': 6, 'name': 'SPLITTER_POA_01_met_kapje'},
    {'supercategory': '', 'id': 7, 'name': 'SPLITTER_POA_01_zonder_kapje'},
    {'supercategory': '', 'id': 8, 'name': 'SPLITTER_POA_01IEC'},
    {'supercategory': '', 'id': 9, 'name': 'SPLITTER_POA_3_met_kapje'},
    {'supercategory': '', 'id': 10, 'name': 'SPLITTER_POA_3_zonder_kapje'},
    {'supercategory': '', 'id': 11, 'name': 'SPLITTER_SQ601_met_kapje'},
    {'supercategory': '', 'id': 12, 'name': 'SPLITTER_UMU_met_kapje'},
    {'supercategory': '', 'id': 13, 'name': 'WCD_tweegats'},
    {'supercategory': '', 'id': 14, 'name': 'AOP_BTV1'},
    {'supercategory': '', 'id': 15, 'name': 'AOP_DIO_01'},
]

# Novel COCO categories
COCO_NOVEL_CATEGORIES = [
    {'supercategory': '', 'id': 2, 'name': 'AOP_TRAS1000'},
    {'supercategory': '', 'id': 7, 'name': 'SPLITTER_POA_01_zonder_kapje'},
]


def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES]
    assert len(thing_ids) == 14, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def _get_coco_fewshot_instances_meta():
    ret = _get_coco_instances_meta()
    novel_ids = [k["id"] for k in COCO_NOVEL_CATEGORIES]
    novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}
    novel_classes = [
        k["name"] for k in COCO_NOVEL_CATEGORIES
    ]
    base_categories = [
        k
        for k in COCO_CATEGORIES
        if k["name"] not in novel_classes
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


def _get_builtin_metadata(dataset_name):
    if dataset_name == "custom":
        return _get_coco_instances_meta()
    elif dataset_name == "custom_fewshot":
        return _get_coco_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
