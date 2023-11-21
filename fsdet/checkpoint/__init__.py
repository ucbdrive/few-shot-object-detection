from . import catalog as _UNUSED  # register the handler
# from .detection_checkpoint import DetectionCheckpointer
from detectron2.checkpoint import DetectionCheckpointer

__all__ = ["DetectionCheckpointer"]