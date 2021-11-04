import os
import sys
from omegaconf import ListConfig

add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)

import torch
import torch.nn.functional as F

from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict


import modules


class TwoMLPHead(nn.Module):

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.class_score = nn.Linear(in_channels, num_classes)
        self.bbox_prediction = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.class_score(x)
        bbox_deltas = self.bbox_prediction(x)

        return scores, bbox_deltas


class TextFuseNet(nn.Module):
    def __init__(self, cfg):
        super(TextFuseNet, self).__init__()

        self.backbone = getattr(modules, cfg['backbone'])(**cfg.get('backbone_args', {}))
        self.neck = modules.FPN(**cfg.get('fpn_args', {}))


        # Anchor Generator
        anchor_scales = cfg.get('anchor_scales', [32, 64, 128, 256, 512])
        aspect_ratios = cfg.get('aspect_ratios', [0.5, 1.0, 2.0])
        anchor_scales = tuple((s,) for s in anchor_scales)
        aspect_ratios = (tuple(aspect_ratios),) * len(anchor_scales)
        rpn_anchor_generator = modules.AnchorGenerator(anchor_scales, aspect_ratios)

        # Region Proposal Network
        self.fpn_out_names = ['p2', 'p3', 'p4', 'p5', 'p6']
        fpn_out_chans = cfg['fpn_args']['mid_chans']
        rpn_head = modules.RPNHead(fpn_out_chans, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=cfg['RPN']['pre_nms_top_n_train'], testing=cfg['RPN']['pre_nms_top_n_test'])
        rpn_post_nms_top_n = dict(training=cfg['RPN']['post_nms_top_n_train'], testing=cfg['RPN']['post_nms_top_n_test'])


        self.rpn = modules.RegionProposalNetwork(
            anchor_generator=rpn_anchor_generator,
            head=rpn_head,
            pre_nms_top_n=rpn_pre_nms_top_n,
            post_nms_top_n=rpn_post_nms_top_n,
            **cfg['RPN']
            )


        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['p2', 'p3', 'p4', 'p5'],
            output_size=7,
            sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            fpn_out_chans * resolution ** 2,
            representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            cfg['num_classes'])


        if isinstance(cfg['transform']['min_size'], ListConfig):
            min_size = list(cfg['transform']['min_size'])
        else:
            min_size = cfg['transform']['min_size']

        if isinstance(cfg['transform']['max_size'], ListConfig):
            max_size = list(cfg['transform']['max_size'])
        else:
            max_size = cfg['transform']['max_size']
        self.transform = modules.GeneralizedRCNNTransform(min_size, max_size)


        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['p2', 'p3', 'p4', 'p5'],
            output_size=14,
            sampling_ratio=2)

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(fpn_out_chans, mask_layers, mask_dilation)

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                            mask_dim_reduced, cfg['num_classes'])


        seg_head = modules.SegHead(**cfg['Seg'])
        multi_fuse_path = modules.MultiFusePath(**cfg['MultiFuse'])


        self.roi_heads = modules.RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_roi_pool=mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor,
            seg_head=seg_head,
            multi_fuse_path=multi_fuse_path,
            fpn_features_fused_level=cfg['MultiFuse']['fpn_features_fused_level'],
            **cfg['Box']
        )

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor

        if cfg['base_pretrained']:
            model_dict = self.state_dict()
            pretrained_dict = torch.load(cfg['base_pretrained'], map_location='cpu')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print(f"{pretrained_dict.keys()}\nLoaded torchvision pretrained weights on COCO.\n")


    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for i, target in enumerate(targets):
                boxes = target["boxes"]
                # print(f'Target-{i} : {boxes}')
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        print(boxes)
                        print(boxes.shape)
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = []

        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        out = self.backbone(images.tensors)
        feature = self.neck(out)
        fpn_outputs = {k:v for k, v in zip(self.fpn_out_names, feature)}
        features = [v for _, v in fpn_outputs.items()]

        if isinstance(feature, torch.Tensor):
            features = OrderedDict([('p2', feature)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(fpn_outputs, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):

        d = OrderedDict()
        next_feature = in_channels

        for layer_idx, layer_features in enumerate(layers, 1):

            d["mask_fcn{}".format(layer_idx)] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)

            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):

        super(MaskRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logit", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


# model_urls = {
#     'maskrcnn_resnet50_fpn_coco':
#         'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',}

if __name__ == "__main__":
    import os
    import os.path as osp
    import sys
    add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(add_dir)
    add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
    sys.path.append(add_dir)
    add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
    sys.path.append(add_dir)

    import yaml
    from pathlib import Path
    from torchinfo import summary

    root = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir, osp.pardir))
    workspace = Path(root) / 'workspace'
    conf_dir  = workspace / 'configs' / 'model'
    conf_name = 'textfusenet_resnet50'
    conf_path = conf_dir / f'{conf_name}.yaml'

    with open(str(conf_path), 'r') as f:
        configs = yaml.safe_load(f)

    model = TextFuseNet(configs)
    model = model.eval()
    summary(model)

    fake_inps = torch.rand((1, 3, 1500, 256))
    fake_outs = model(fake_inps)
    losses, detections = fake_outs

    print('Losses:\n', losses)
    print('Detections:\n')
    for key, value in detections[0].items():
        print(f'{key}: {value.shape}\n')
        print(value, '\n')