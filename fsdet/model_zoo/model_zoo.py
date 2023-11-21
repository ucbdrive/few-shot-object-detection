import torch

from fsdet.modeling import build_model

import os
import pkg_resources
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg


class _ModelZooUrls(object):
    """
    Mapping from names to our pre-trained models.
    """

    URL_PREFIX = "http://dl.yf.io/fs-det/models/"

    # format: {config_path.yaml} -> model_id/model_final.pth
    CONFIG_PATH_TO_URL_SUFFIX = {
        ### PASCAL VOC Detection ###
        # Base Model
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml": "voc/split1/base_model/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_base2.yaml": "voc/split2/base_model/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_base3.yaml": "voc/split3/base_model/model_final.pth",
        # FRCN+ft-full
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot_unfreeze.yaml": "voc/split1/FRCN+ft-full_1shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot_unfreeze.yaml": "voc/split1/FRCN+ft-full_2shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_3shot_unfreeze.yaml": "voc/split1/FRCN+ft-full_3shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot_unfreeze.yaml": "voc/split1/FRCN+ft-full_5shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot_unfreeze.yaml": "voc/split1/FRCN+ft-full_10shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_1shot_unfreeze.yaml": "voc/split2/FRCN+ft-full_1shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_2shot_unfreeze.yaml": "voc/split2/FRCN+ft-full_2shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_3shot_unfreeze.yaml": "voc/split2/FRCN+ft-full_3shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot_unfreeze.yaml": "voc/split2/FRCN+ft-full_5shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot_unfreeze.yaml": "voc/split2/FRCN+ft-full_10shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_1shot_unfreeze.yaml": "voc/split3/FRCN+ft-full_1shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_2shot_unfreeze.yaml": "voc/split3/FRCN+ft-full_2shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_3shot_unfreeze.yaml": "voc/split3/FRCN+ft-full_3shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_5shot_unfreeze.yaml": "voc/split3/FRCN+ft-full_5shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot_unfreeze.yaml": "voc/split3/FRCN+ft-full_10shot/model_final.pth",
        # TFA w/ cos
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml": "voc/split1/tfa_cos_1shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml": "voc/split1/tfa_cos_2shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_3shot.yaml": "voc/split1/tfa_cos_3shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml": "voc/split1/tfa_cos_5shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml": "voc/split1/tfa_cos_10shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_1shot.yaml": "voc/split2/tfa_cos_1shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_2shot.yaml": "voc/split2/tfa_cos_2shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_3shot.yaml": "voc/split2/tfa_cos_3shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot.yaml": "voc/split2/tfa_cos_5shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml": "voc/split2/tfa_cos_10shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_1shot.yaml": "voc/split3/tfa_cos_1shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_2shot.yaml": "voc/split3/tfa_cos_2shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_3shot.yaml": "voc/split3/tfa_cos_3shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_5shot.yaml": "voc/split3/tfa_cos_5shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot.yaml": "voc/split3/tfa_cos_10shot/model_final.pth",
        # TFA w/ fc
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_1shot.yaml": "voc/split1/tfa_fc_1shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_2shot.yaml": "voc/split1/tfa_fc_2shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_3shot.yaml": "voc/split1/tfa_fc_3shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml": "voc/split1/tfa_fc_5shot/model_final.pth",
        "PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_10shot.yaml": "voc/split1/tfa_fc_10shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_1shot.yaml": "voc/split2/tfa_fc_1shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_2shot.yaml": "voc/split2/tfa_fc_2shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_3shot.yaml": "voc/split2/tfa_fc_3shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_5shot.yaml": "voc/split2/tfa_fc_5shot/model_final.pth",
        "PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_fc_all2_10shot.yaml": "voc/split2/tfa_fc_10shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_1shot.yaml": "voc/split3/tfa_fc_1shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_2shot.yaml": "voc/split3/tfa_fc_2shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_3shot.yaml": "voc/split3/tfa_fc_3shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_5shot.yaml": "voc/split3/tfa_fc_5shot/model_final.pth",
        "PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_fc_all3_10shot.yaml": "voc/split3/tfa_fc_10shot/model_final.pth",
        ### COCO Detection ###
        # Base Model
        "COCO-detection/faster_rcnn_R_101_FPN_base.yaml": "coco/base_model/model_final.pth",
        # FRCN+ft-full
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot_unfreeze.yaml": "coco/FRCN+ft-full_1shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_2shot_unfreeze.yaml": "coco/FRCN+ft-full_2shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_3shot_unfreeze.yaml": "coco/FRCN+ft-full_3shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_5shot_unfreeze.yaml": "coco/FRCN+ft-full_5shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot_unfreeze.yaml": "coco/FRCN+ft-full_10shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot_unfreeze.yaml": "coco/FRCN+ft-full_30shot/model_final.pth",
        # TFA w/ cos
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml": "coco/tfa_cos_1shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_2shot.yaml": "coco/tfa_cos_2shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_3shot.yaml": "coco/tfa_cos_3shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_5shot.yaml": "coco/tfa_cos_5shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml": "coco/tfa_cos_10shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_all_30shot.yaml": "coco/tfa_cos_30shot/model_final.pth",
        # TFA w/ fc
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml": "coco/tfa_fc_1shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_2shot.yaml": "coco/tfa_fc_2shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_3shot.yaml": "coco/tfa_fc_3shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml": "coco/tfa_fc_5shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml": "coco/tfa_fc_10shot/model_final.pth",
        "COCO-detection/faster_rcnn_R_101_FPN_ft_fc_all_30shot.yaml": "coco/tfa_fc_30shot/model_final.pth",
        ### LVIS Detection ###
        # Base Models
        ## With repeat sampling
        "LVIS-detection/faster_rcnn_R_50_FPN_base.yaml": "lvis/R_50_FPN_base_repeat_fc/model_final.pth",
        "LVIS-detection/faster_rcnn_R_50_FPN_base_cosine.yaml": "lvis/R_50_FPN_base_repeat_cos/model_final.pth",
        "LVIS-detection/faster_rcnn_R_101_FPN_base.yaml": "lvis/R_101_FPN_base_repeat_fc/model_final.pth",
        "LVIS-detection/faster_rcnn_R_101_FPN_base_cosine.yaml": "lvis/R_101_FPN_base_repeat_cos/model_final.pth",
        ## No repeat sampling
        "LVIS-detection/faster_rcnn_R_50_FPN_base_norepeat.yaml": "lvis/R_50_FPN_base_norepeat_fc/model_final.pth",
        "LVIS-detection/faster_rcnn_R_50_FPN_base_norepeat_cosine.yaml": "lvis/R_50_FPN_base_norepeat_cos/model_final.pth",
        "LVIS-detection/faster_rcnn_R_101_FPN_base_norepeat.yaml": "lvis/R_101_FPN_base_norepeat_fc/model_final.pth",
        "LVIS-detection/faster_rcnn_R_101_FPN_base_norepeat_cosine.yaml": "lvis/R_101_FPN_base_norepeat_cos/model_final.pth",
        # Fine-tuned Models
        ## With repeat sampling
        "LVIS-detection/faster_rcnn_R_50_FPN_combined_all.yaml": "lvis/R_50_FPN_repeat_fc/model_final.pth",
        "LVIS-detection/faster_rcnn_R_50_FPN_cosine_combined_all.yaml": "lvis/R_50_FPN_repeat_cos/model_final.pth",
        "LVIS-detection/faster_rcnn_R_101_FPN_combined_all.yaml": "lvis/R_101_FPN_repeat_fc/model_final.pth",
        "LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all.yaml": "lvis/R_101_FPN_repeat_cos/model_final.pth",
        ## No repeat sampling
        "LVIS-detection/faster_rcnn_R_50_FPN_combined_all_norepeat.yaml": "lvis/R_50_FPN_norepeat_fc/model_final.pth",
        "LVIS-detection/faster_rcnn_R_50_FPN_cosine_combined_all_norepeat.yaml": "lvis/R_50_FPN_norepeat_cos/model_final.pth",
        "LVIS-detection/faster_rcnn_R_101_FPN_combined_all_norepeat.yaml": "lvis/R_101_FPN_norepeat_fc/model_final.pth",
        "LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml": "lvis/R_101_FPN_norepeat_cos/model_final.pth",
    }


def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config
    Args:
        config_path (str): config file name relative to FsDet's "configs/"
            directory, e.g., "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
    Returns:
        str: a URL to the model
    """
    if config_path in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
        suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[config_path]
        return _ModelZooUrls.URL_PREFIX + suffix
    raise RuntimeError("{} not available in Model Zoo!".format(config_path))


def get_config_file(config_path):
    """
    Returns path to a builtin config file.
    Args:
        config_path (str): config file name relative to FsDet's "configs/"
            directory, e.g., "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "fsdet.model_zoo", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError(
            "{} not available in Model Zoo!".format(config_path)
        )
    return cfg_file


def get(config_path, trained: bool = False):
    """
    Get a model specified by relative path under FsDet's official ``configs/`` directory.
    Args:
        config_path (str): config file name relative to FsDet's "configs/"
            directory, e.g., "COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
        trained (bool): If True, will initialize the model with the trained model zoo weights.
            If False, the checkpoint specified in the config file's ``MODEL.WEIGHTS`` is used
            instead; this will typically (though not always) initialize a subset of weights using
            an ImageNet pre-trained model, while randomly initializing the other weights.
    Example:
    .. code-block:: python
        from fsdet import model_zoo
        model = model_zoo.get("COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml", trained=True)
    """
    cfg_file = get_config_file(config_path)

    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    if trained:
        cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model