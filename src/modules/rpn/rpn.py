import os
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from typing import List, Tuple, Dict
from prettyprinter import pprint, cpprint

from src.modules.rpn.anchor_generator import AnchorGenerator
from src.modules.utils import BoxCoder, Matcher, BalancedPositiveNegativeSampler



class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):

        super(RPNHead, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)


    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []

        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))

        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)

    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression):

        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A

        # print(f'obj previous : {box_cls_per_level.shape}')
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        # print(f'obj after : {box_cls_per_level.shape}')
        box_cls_flattened.append(box_cls_per_level)

        # print(f'bbox previous : {box_regression_per_level.shape}')
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        # print(f'bbox after : {box_regression_per_level.shape}')
        # print('\n')
        box_regression_flattened.append(box_regression_per_level)

    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    # print(f'Box class shape : {box_cls.shape}')
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    # print(f'Box regression shape : {box_regression.shape}')

    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).
    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
    """

    def __init__(self,
                 anchor_generator,
                 head,
                 fg_iou_thresh,
                 bg_iou_thresh,
                 batch_size_per_image,
                 positive_fraction,
                 pre_nms_top_n,
                 post_nms_top_n,
                 nms_thresh,
                 score_thresh=0.0, **kwrags):

        super(RegionProposalNetwork, self).__init__()

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3


    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']


    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):

            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            # offset : Variables to Differentiate Pyramid Levels
            r.append(top_n_idx + offset)
            offset += num_anchors

        return torch.cat(r, dim=1)


    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):

        num_images = proposals.shape[0]
        device = proposals.device

        ## do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)
        # objectness shape : [N, 12543]

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        # levels = tensor([0, 0, 0, ..., 4, 4, 4]) / shape : [12543]
        levels = levels.reshape(1, -1).expand_as(objectness)
        # levels shape : [N, 12543]

        ## select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        # print(top_n_idx.shape)
        # [2, 9135]

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        # tensor([0, 1]) ->
        # tensor([[0],
        #         [1]])

        objectness = objectness[batch_idx, top_n_idx] # [2, 9135]
        levels = levels[batch_idx, top_n_idx] # [2, 9135]
        proposals = proposals[batch_idx, top_n_idx] # [2, 9135, 4]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []

        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # clip box; boxes in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores


    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction='sum',
        ) / (sampled_inds.numel())

        # box_loss = smooth_l1_loss(
        #     pred_bbox_deltas[sampled_pos_inds],
        #     regression_targets[sampled_pos_inds],
        #     beta=1 / 9
        # )

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss


    def forward(self,
                images,
                features,
                targets=None
                ):

        ## RPN uses all feature maps that are available
        objectness, pred_bbox_deltas = self.head(features)

        ## RPN HEAD Output Shape
        # for i, (obj, delta) in enumerate(zip(objectness, pred_bbox_deltas)):
        #     cpprint(f'P{i + 2} objectness shape : {obj.shape}')
        #     cpprint(f'P{i + 2} bbox_reg shape : {delta.shape}')
        #     print('\n')
        #
        # P2 objectness shape : torch.Size([1, 3, 56, 56])
        # P3 objectness shape : torch.Size([1, 3, 28, 28])
        # P4 objectness shape : torch.Size([1, 3, 14, 14])
        # P5 objectness shape : torch.Size([1, 3, 7, 7])
        # P6 objectness shape : torch.Size([1, 3, 4, 4])

        # P2 bbox shape : torch.Size([1, 12, 56, 56])
        # P3 bbox shape : torch.Size([1, 12, 28, 28])
        # P4 bbox shape : torch.Size([1, 12, 14, 14])
        # P5 bbox shape : torch.Size([1, 12, 7, 7])
        # P6 bbox shape : torch.Size([1, 12, 4, 4])

        anchors = self.anchor_generator(images, features)
        ## Anchor shape / image_size : 224 x 224
        # cpprint(Tensor(*anchors).shape)
        # torch.Size([12543, 4])
        # P2(56 x 56 x 3) + P3(28 x 28 x 3) + P4(14 x 14 x 3) + P5(7 x7 x 3) + P6(4 x 4 + 3) = 12543

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        ## Output Shape
        # objectness shape : [12543, 1]
        # pred_bbox_deltas shape : [12543, 4]


        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        # output shape : [1, 12543, 4]
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # Filtering
        # pre-nms top k, clip boxes, removal of invalid boxes, nms, post-nms top N
        # concatenate boxes from all feature maps of FPN
        boxes, _ = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        # Compute Loss
        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        return boxes, losses


if __name__ == "__main__":

    ## RPN Head Unit Test
    from models.neck.fpn import BackboneWithFPN
    from utils.det_utils import ImageList

    images = torch.randn((2, 3, 224, 224))
    backbone_with_fpn = BackboneWithFPN('resnet50d', True, 256, False, False)
    output = backbone_with_fpn(images)

    print('\n\n' + '=' * 15, 'Backbone Ouput Information', '=' * 15)
    for idx, (k, v) in enumerate(output.items()):
        cpprint(f'feature_map_name : {k}')
        cpprint(f'feature_map_size : {v.shape}')
        print('-' * 58)
    print('=' * 58, '\n\n')

    print(output.keys())

    features = [v for _, v in output.items()]

    # RPN parameters
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    logits, bbox_reg = head(features)

    for i, logit in enumerate(logits):
        print(f'P{i + 2} Logit shape : {logit.shape}')
    print('\n')

    for i, bbox in enumerate(bbox_reg):
        print(f'P{i + 2} Bbox shape : {bbox.shape}')
    print('\n')

    ## RPN Unit Test
    # generate dummy data
    image_sizes_list = []
    image_sizes = [img.shape[-2:] for img in images]

    for image_size in image_sizes:
        image_sizes_list.append((image_size[0], image_size[1]))
    image_list = ImageList(images, image_sizes_list)

    pre_nms_top_n = {'training' : 12000, 'testing' : 6000}
    post_nms_top_n = {'training' : 2000, 'testing' : 1000}

    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3
    batch_size_per_image = 256
    positive_fraction = 0.5
    nms_thresh = 0.7

    rpn = RegionProposalNetwork(anchor_generator=anchor_generator,
                                pre_nms_top_n=pre_nms_top_n,
                                post_nms_top_n=post_nms_top_n,
                                head=head,
                                fg_iou_thresh=fg_iou_thresh,
                                bg_iou_thresh=bg_iou_thresh,
                                batch_size_per_image=batch_size_per_image,
                                positive_fraction=positive_fraction,
                                nms_thresh=nms_thresh)

    rpn.training = False

    boxes, _ = rpn(image_list, features)

    print(f'Output : {boxes[0].shape}')