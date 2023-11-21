"""Implement the CosineSimOutputLayers and  FastRCNNOutputLayers with FC layers."""
import numpy as np
import torch
import math
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

import logging
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

ROI_HEADS_OUTPUT_REGISTRY = Registry("ROI_HEADS_OUTPUT")
ROI_HEADS_OUTPUT_REGISTRY.__doc__ = """
Registry for the output layers in ROI heads in a generalized R-CNN model."""

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
    boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """

    # if not isinstance(scores, list):
    #     result_per_image = [
    #         fast_rcnn_inference_single_image(
    #             boxes_per_image,
    #             scores_per_image,
    #             image_shape,
    #             score_thresh,
    #             nms_thresh,
    #             topk_per_image,
    #         )
    #         for scores_per_image, boxes_per_image, image_shape in zip(
    #             scores, boxes, image_shapes
    #         )
    #     ]
    #     return tuple(list(x) for x in zip(*result_per_image))
    #
    # else:
    #     result_per_image = []
    #     fast_rcnn_inference_single_image_ensemble(scores,
    #                 boxes,
    #                 image_shapes,
    #                 score_thresh,
    #                 nms_thresh,
    #                 topk_per_image,)
    #     return tuple(list(x) for x in zip(*result_per_image))

    if isinstance(scores, list):
        score_ens = list(scores[0])
        for score in list(scores[1:]):
            for img_idx in range(len(image_shapes)):
                score_ens[img_idx] = torch.cat((score_ens[img_idx], score[img_idx]), dim=0)
        score_ens = tuple(score_ens)

        boxes_ens = list(boxes)
        for head_idx in range(1, len(scores)):
            for img_idx in range(len(image_shapes)):
                boxes_ens[img_idx] = torch.cat((boxes_ens[img_idx], boxes[img_idx]), dim=0)
        boxes_ens = tuple(boxes_ens)

        scores = score_ens
        boxes = boxes_ens

    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(
            scores, boxes, image_shapes
        )
    ]
    return tuple(list(x) for x in zip(*result_per_image))

def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    # filter_inds = filter_mask.nonzero()
    filter_inds = torch.nonzero(filter_mask, as_tuple=False)
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def fast_rcnn_inference_single_image_ensemble(
    scores, boxes, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    score_ens = list(scores[0])
    for score in list(scores[1:]):
        for img_idx in range(len(image_shapes)):
            score_ens[img_idx] = torch.cat((score_ens[img_idx], score[img_idx]), dim=0)
    score_ens = tuple(score_ens)

    boxes_ens = list(boxes)
    for head_idx in range(1,len(scores)):
        for img_idx in range(len(image_shapes)):
            boxes_ens[img_idx] = torch.cat((boxes_ens[img_idx], boxes[img_idx]), dim=0)
    boxes_ens = tuple(boxes_ens)

    score_ens = score_ens[:, :-1]
    num_bbox_reg_classes = boxes_ens.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes_ens = Boxes(boxes_ens.reshape(-1, 4))
    boxes_ens.clip(image_shapes)
    boxes_ens = boxes_ens.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = score_ens > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes_ens = boxes_ens[filter_inds[:, 0], 0]
    else:
        boxes_ens = boxes_ens[filter_mask]
    score_ens = score_ens[filter_mask]

    result = Instances(image_shapes)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert (
            not self.proposals.tensor.requires_grad
        ), "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        # pred_classes = self.pred_class_logits[0].argmax(dim=1)
        # bg_class_ind = self.pred_class_logits[0].shape[0] - 1
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (
            (fg_pred_classes == bg_class_ind).nonzero().numel()
        )
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar(
            "fast_rcnn/cls_accuracy", num_accurate / num_instances
        )
        if num_fg > 0:
            storage.put_scalar(
                "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
            )
            storage.put_scalar(
                "fast_rcnn/false_negative", num_false_negative / num_fg
            )

    def ensemble_loss(self):
        total_loss, total_acc = 0, 0
        for logits in self.pred_class_logits:
            if logits is not None:
                log_p_y = F.log_softmax(logits, dim=1)
                preds = log_p_y.argmax(-1)
                labels = self.gt_classes.type(torch.long)
                loss = F.nll_loss(log_p_y, labels, reduction='mean')
                acc = torch.eq(preds, labels).float().mean()
                total_loss = total_loss + loss
                total_acc = total_acc + acc
        return total_loss

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        
        self._log_accuracy()
        
        use_ovss = False
        
        reductionmethod = "mean"
        if (use_ovss):
            reductionmethod = "none"
        
        smceloss = F.cross_entropy(
            self.pred_class_logits, self.gt_classes, reduction=reductionmethod
        )
          
        if (use_ovss):
            device = self.gt_boxes.device

            ovss_w = torch.tensor(self.get_ovss_weights(self.gt_boxes.tensor,self.proposals),device=device)
          
            smceloss.register_hook(lambda grad, weights=ovss_w: ovss_grad(grad,weights) )            
            
            smceloss = smceloss.mean()    
            
        return smceloss  
        

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        if isinstance(self.pred_class_logits, list):
            bg_class_ind = self.pred_class_logits[0].shape[1]-1
        else:
            bg_class_ind = self.pred_class_logits.shape[1] - 1



        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        ).squeeze(1)
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )


        use_ovss = False
        
        reductionmethod = "sum"
        if (use_ovss):
            reductionmethod = "none"

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction=reductionmethod, 
        )
        
        if (use_ovss):

            ovss_w = torch.tensor(self.get_ovss_weights(self.gt_boxes.tensor[fg_inds],self.proposals[fg_inds]),device=device)
          
            loss_box_reg.register_hook(lambda grad, weights=ovss_w: ovss_grad(grad,weights) )            
            
            loss_box_reg = loss_box_reg.sum()

           

        
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        l1loss = self.smooth_l1_loss()
        #print(l1loss)

        if isinstance(self.pred_class_logits, list):
            return {
                # "loss_cls": self.softmax_cross_entropy_loss(),
                "loss_cls": self.ensemble_loss(),
                "loss_box_reg": l1loss,
            }

        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": l1loss,
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1)
            .expand(num_pred, K, B)
            .reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(
            self.num_preds_per_image, dim=0
        )

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        # Add ensemble logic here
        if not isinstance(self.pred_class_logits, list):
            probs = F.softmax(self.pred_class_logits, dim=-1)
            return probs.split(self.num_preds_per_image, dim=0)
        else:
            # perform mean over all head outputs
            # probs = torch.mean(torch.stack(self.pred_class_logits), dim=0)
            probs = [F.softmax(class_logit, dim=-1) for class_logit in self.pred_class_logits]
            return [prob.split(self.num_preds_per_image, dim=0) for prob in probs]

        # return probs.split(self.num_preds_per_image, dim=0)


    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )


    def get_ovss_weights(self,gt,pred):
        a = 0.25
        b = 50
        c = 20
    
        w = np.zeros((gt.shape[0],))
    
        for i in range(gt.shape[0]):
    
            assert(pred[i].tensor.shape[0]==1)
    
            iou = get_iou(gt[i], pred[i].tensor[0])
            
            w[i] = a + (1-a) * math.exp(-b * math.exp(-c*iou))
            
           
        return w
            

@ROI_HEADS_OUTPUT_REGISTRY.register()
class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
        self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas


@ROI_HEADS_OUTPUT_REGISTRY.register()
class CosineSimOutputLayers(nn.Module):
    """
    Two outputs
    (1) proposal-to-detection box regression deltas (the same as
        the FastRCNNOutputLayers)
    (2) classification score is based on cosine_similarity
    """

    def __init__(
        self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(CosineSimOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1, bias=False)
        self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        if self.scale == -1:
            # learnable global scaling factor
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas
      
        
def ovss_grad(grad,weights):

    weights = weights.float()

    if len(grad.shape)==1:
        grad = grad * weights
        return grad

    for i in range(len(weights)):
        for j in range(grad.shape[1]):
            grad[i,j] = grad[i,j]*weights[i]

    return grad
    
def get_iou(bb1, bb2):
    # format of each bb: x1,y1,x2,y2

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    bb2_area = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


@ROI_HEADS_OUTPUT_REGISTRY.register()
class EnsembledFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
            self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(EnsembledFastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.num_heads = cfg.MODEL.ROI_HEADS.NUM_HEADS
        self.head_drop_prob = cfg.MODEL.ROI_HEADS.HEAD_DROP_PROB
        # self.cls_score = [nn.Linear(input_size, num_classes + 1) for x in range(self.num_heads)]
        self.cls_score = nn.ModuleList([nn.Linear(input_size, num_classes + 1) for x in range(self.num_heads)])
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        for score in self.cls_score:
            nn.init.normal_(score.weight, std=0.01)
            nn.init.constant_(score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def sample_bool_inds(self, prob=0.7):
        bool_inds = (torch.rand(self.num_heads) < prob).type(torch.uint8)
        if bool_inds.type(torch.int).sum() == 0:
            rnd_ind = (torch.rand(1) * self.num_heads).type(torch.long)
            bool_inds[rnd_ind] = 1
        return bool_inds

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        outs = []
        bool_inds = self.sample_bool_inds(self.head_drop_prob)
        for i in range(self.num_heads):
            if not bool_inds[i] and self.training:
                continue
            self.cls_score[i].to(device=x.device)
            out = self.cls_score[i](x)
            outs.append(out)

        # output = sum(outs) / len(outs) if avg else outs

        proposal_deltas = self.bbox_pred(x)
        return outs, proposal_deltas


