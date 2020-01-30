import torch
import torch.nn as nn
from mmdet.core.evaluation import bbox_overlaps
from ..registry import LOSSES
from .utils import weighted_loss
import math
import torch.nn.functional as F


def IoG(box_a,box_b):
    inter_xmin=torch.max(box_a[0],box_b[0])
    inter_ymin=torch.max(box_a[1],box_b[1])
    inter_xmax=torch.min(box_a[2],box_b[2])
    inter_ymax=torch.min(box_a[3],box_b[3])
    Iw=torch.clamp(inter_xmax-inter_xmin,min=0)
    Ih=torch.clamp(inter_ymax-inter_ymin,min=0)
    I=Iw*Ih
    G=(box_b[2]-box_b[0])*(box_b[3]-box_b[1])
    return I/G

def bbox_overlaps_(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)
    #import pdb
    #pdb.set_trace()
    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()



@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


def repgt_loss(pred,target,reduction='mean',avg_factor=None):
    sigma_repgt = 0.9
    loss_repgt = torch.zeros(pred.shape[0]).cuda()
    for i in range(pred.shape[0]):
        # boxes = Variable(
        #     pred_boxes[i, rois_inside_ws[i] != 0].view(int(pred_boxes[i, rois_inside_ws[i] != 0].shape[0]) / 4, 4))
        # gt = Variable(gt_rois[i, rois_inside_ws[i] != 0].view(int(gt_rois[i, rois_inside_ws[i] != 0].shape[0]) / 4, 4))
        num_repgt = 0
        repgt_smoothln = 0
        if pred.shape[0] > 0:
            overlaps = bbox_overlaps_(pred, target)
            for j in range(overlaps.shape[0]):
                for z in range(overlaps.shape[1]):
                    if int(torch.sum(target[j] == target[z])) == 4:
                        overlaps[j, z] = 0
            max_overlaps, argmax_overlaps = torch.max(overlaps, 1)
            for j in range(max_overlaps.shape[0]):
                if max_overlaps[j] > 0:
                    num_repgt += 1
                    iog = IoG(pred[j], target[argmax_overlaps[j]])
                    if iog > sigma_repgt:
                        repgt_smoothln += ((iog - sigma_repgt) / (1 - sigma_repgt) - math.log(1 - sigma_repgt))
                    elif iog <= sigma_repgt:
                        repgt_smoothln += -math.log(1 - iog)
        if num_repgt > 0:
            loss_repgt[i] = repgt_smoothln / num_repgt
    if avg_factor is None:
        loss = reduce_loss(loss_repgt, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss_repgt.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss_repgt


def repbox_loss(pred,target,reduction='mean',avg_factor=None):
    sigma_repbox = 0
    loss_repbox = torch.zeros(pred.shape[0]).cuda()

    for i in range(pred.shape[0]):

        # boxes = Variable(
        #     pred_boxes[i, rois_inside_ws[i] != 0].view(int(pred_boxes[i, rois_inside_ws[i] != 0].shape[0]) / 4, 4))
        # gt = Variable(gt_rois[i, rois_inside_ws[i] != 0].view(int(gt_rois[i, rois_inside_ws[i] != 0].shape[0]) / 4, 4))

        num_repbox = 0
        repbox_smoothln = 0
        if pred.shape[0] > 0:
            overlaps = bbox_overlaps_(pred, target)
            for j in range(overlaps.shape[0]):
                for z in range(overlaps.shape[1]):
                    if z >= j:
                        overlaps[j, z] = 0
                    elif int(torch.sum(target[j] == target[z])) == 4:
                        overlaps[j, z] = 0

            iou = overlaps[overlaps > 0]
            for j in range(iou.shape[0]):
                num_repbox += 1
                if iou[j] <= sigma_repbox:
                    repbox_smoothln += -math.log(1 - iou[j])
                elif iou[j] > sigma_repbox:
                    repbox_smoothln += ((iou[j] - sigma_repbox) / (1 - sigma_repbox) - math.log(1 - sigma_repbox))

        if num_repbox > 0:
            loss_repbox[i] = repbox_smoothln / num_repbox
    if avg_factor is None:
        loss = reduce_loss(loss_repbox, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss_repbox.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss_repbox






@LOSSES.register_module
class RepulsionLoss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(RepulsionLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)+self.loss_weight*repgt_loss(pred,target,reduction=reduction,avg_factor=avg_factor)\
                    +self.loss_weight*repbox_loss(pred,target,reduction=reduction,avg_factor=avg_factor)
        return loss_bbox
