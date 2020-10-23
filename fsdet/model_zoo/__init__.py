"""
Model Zoo API for FsDet: a collection of functions to create common model architectures and
optionally load pre-trained weights as released in
`MODEL_ZOO.md <https://github.com/ucbdrive/few-shot-object-detection/blob/master/MODEL_ZOO.md>`_.
"""
from .model_zoo import get, get_checkpoint_url, get_config_file

__all__ = ["get_checkpoint_url", "get", "get_config_file"]
