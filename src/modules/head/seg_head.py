import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign


class SegHead(nn.Module):
    def __init__(self, chans,
                       num_feats, num_conv3x3, num_classes,
                       pooler_resolution, pooler_scales, sampling_ratio, pooler_type):
        super(SegHead, self).__init__()

        self.chans = chans
        self.num_feats = num_feats
        self.num_conv3x3 = num_conv3x3
        self.pooler_resolution = pooler_resolution
        self.pooler_scales = pooler_scales
        self.sampling_ratio = sampling_ratio
        self.pooler_type = pooler_type

        self.num_classes = num_classes
        self.conv1x1_list = nn.ModuleDict()
        for key in self.num_feats:
            self.conv1x1_list[key] = nn.Conv2d(self.chans, self.chans, 1, padding=1, bias=False)

        self.conv3x3_list = nn.ModuleList()
        for i in range(self.num_conv3x3):
            self.conv3x3_list.append(nn.Conv2d(self.chans, self.chans, 3, padding=1, bias=False))

        self.seg_pooler = RoIAlign(
            output_size=self.pooler_resolution,
            spatial_scale=self.pooler_scales,
            sampling_ratio=self.sampling_ratio,
            aligned=True
        )

        self.conv3x3_list_roi = nn.ModuleList()
        for i in range(self.num_conv3x3):
            self.conv3x3_list_roi.append(nn.Conv2d(self.chans, self.chans, 3, padding=1, bias=False))


        # layers---segmentation logits
        self.conv1x1_seg_logits = nn.Conv2d(self.chans, self.chans, 1, padding=0, bias=False)
        self.seg_logits = nn.Conv2d(self.chans, self.num_classes, 1)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, feature_level, proposal_boxes, image_shape):
        feature_shape = x[feature_level].shape[-2:]
        feature_fuse = self.conv1x1_list[feature_level](x[feature_level])

        # get global fused features
        for feat_name in self.num_feats:
            if feat_name != feature_level:
                feature = x[feat_name]
                feature = F.interpolate(feature, size=feature_shape, mode='bilinear', align_corners=True)
                feature_fuse += self.conv1x1_list[feat_name](feature)

        for i in range(self.num_conv3x3):
            feature_fuse = self.conv3x3_list[i](feature_fuse)


        # get global context
        global_context = self.seg_pooler(feature_fuse, proposal_boxes)
        for i in range(self.num_conv3x3):
            global_context = self.conv3x3_list_roi[i](global_context)
        global_context = self.relu(global_context)


        # get segmentation logits
        feature_pred = F.interpolate(feature_fuse, size=image_shape, mode='bilinear', align_corners=True)
        feature_pred = self.conv1x1_seg_logits(feature_pred)
        seg_logits = self.seg_logits(feature_pred)

        return seg_logits, global_context



def make_segmentation_gt(targets, shape):
    # W = targets[0]['shape'][1]
    # H = targets[0]['shape'][0]
    W, H = shape
    seg_gts = []

    for i in range(len(targets)):
        classes = targets[i]['labels']
        word_indx = (classes==1).nonzero()

        gt_polygon_list = targets[i]['polygons']

        imglist = []
        for i in word_indx:
            point = gt_polygon_list[i].detach().cpu().numpy()
            point = np.array(point, dtype=np.int32)
            img = np.zeros([W, H], dtype="uint8")
            imglist.append(torch.Tensor(cv2.fillPoly(img, [point], (1, 1, 1))))

        imglist = torch.stack(imglist)
        segmentation_gt = torch.sum(imglist, dim=0)
        segmentation_gt = segmentation_gt > 0
        segmentation_gt = segmentation_gt.reshape(1, W, H)
        segmentation_gt = segmentation_gt.cuda()
        seg_gts.append(segmentation_gt)

    return torch.cat(seg_gts)


class SegLoss(nn.Module):
    """
    compute seg loss
    """
    def __init__(self):
        super(SegLoss, self).__init__()

        self.weight = 0.2
        self.loss = nn.CrossEntropyLoss()

    def forward(self, seg_logits, targets):
        seg_gt = make_segmentation_gt(targets, seg_logits.shape[-2:])
        loss_seg = self.weight * self.loss(seg_logits, seg_gt.long())

        return loss_seg