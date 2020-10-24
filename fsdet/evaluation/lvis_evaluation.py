import copy
import itertools
import json
import logging
import os
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import create_small_table

from fsdet.evaluation.coco_evaluation import instances_to_coco_json
from fsdet.evaluation.evaluator import DatasetEvaluator


class LVISEvaluator(DatasetEvaluator):
    """
    Evaluate instance detection outputs using LVIS's metrics and evaluation API.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                    "json_file": the path to the LVIS format annotation
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        from lvis import LVIS

        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._lvis_api = LVIS(json_file)
        # Test set json files do not contain annotations (evaluation must be
        # performed using the LVIS evaluation server).
        self._do_evaluation = len(self._lvis_api.get_ann_ids()) > 0

    def reset(self):
        self._predictions = []
        self._lvis_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a LVIS model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a LVIS model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return

        if len(self._predictions) == 0:
            self._logger.warning(
                "[LVISEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results in the LVIS format ...")
        self._lvis_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for LVIS
        if hasattr(self._metadata, "class_mapping"):
            # using reverse mapping
            reverse_id_mapping = {
                v: k for k, v in self._metadata.class_mapping.items()
            }
            for result in self._lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]] + 1
        else:
            # from 0-indexed to 1-indexed
            for result in self._lvis_results:
                result["category_id"] += 1

        if self._output_dir:
            file_path = os.path.join(
                self._output_dir, "lvis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._lvis_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        res = _evaluate_predictions_on_lvis(
            self._lvis_api, self._lvis_results, "bbox",
            class_names=self._metadata.get("thing_classes"),
        )
        self._results["bbox"] = res


def _evaluate_predictions_on_lvis(
    lvis_gt, lvis_results, iou_type, class_names=None):
    """
    Args:
        iou_type (str):
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """
    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"]

    logger = logging.getLogger(__name__)

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model! Set scores to -1")
        return {metric: -1 for metric in metrics}

    from lvis import LVISEval, LVISResults

    lvis_results = LVISResults(lvis_gt, lvis_results)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    logger.info(
        "Evaluation results for {}: \n".format(iou_type) + \
            create_small_table(results)
    )
    return results
