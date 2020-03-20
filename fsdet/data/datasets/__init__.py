# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .coco import load_coco_json
from .lvis import load_lvis_json, register_lvis_instances
from .register_coco import register_coco_instances
from . import builtin  # ensure the builtin datasets are registered


__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
