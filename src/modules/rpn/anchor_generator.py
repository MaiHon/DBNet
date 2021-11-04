import torch

from torch import nn, Tensor
from prettyprinter import pprint, cpprint
from typing import List, Optional, Dict


class AnchorGenerator(nn.Module):

    def __init__(
        self,
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super(AnchorGenerator, self).__init__()

        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)

        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}


    def generate_anchors(self, scales, aspect_ratios, dtype = torch.float32,
                         device = torch.device("cpu")):

        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)


        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        # pprint(base_anchors.round())

        ## Generate Anchor from Original Image Size
        ## Original paper Image size : VGG(224 x 224) / ResNet (N x N); min(N) -> 600, max(N) -> 1024
        # (32,)
        # tensor([[-23., -11.,  23.,  11.],
        #         [-16., -16.,  16.,  16.],
        #         [-11., -23.,  11.,  23.]])
        # (64,)
        # tensor([[-45., -23.,  45.,  23.],
        #         [-32., -32.,  32.,  32.],
        #         [-23., -45.,  23.,  45.]])
        # (128,)
        # tensor([[-91., -45.,  91.,  45.],
        #         [-64., -64.,  64.,  64.],
        #         [-45., -91.,  45.,  91.]])
        # (256,)
        # tensor([[-181.,  -91.,  181.,   91.],
        #         [-128., -128.,  128.,  128.],
        #         [ -91., -181.,   91.,  181.]])
        # (512,)
        # tensor([[-362., -181.,  362.,  181.],
        #         [-256., -256.,  256.,  256.],
        #         [-181., -362.,  181.,  362.]])

        return base_anchors.round()


    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None

            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(
                sizes,
                aspect_ratios,
                dtype,
                device
            )
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors


    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]


    def grid_anchors(self, grid_sizes, strides):

        anchors = []
        cell_anchors = self.cell_anchors

        assert cell_anchors is not None

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):

            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x) # make grid
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        ## Anchors at each pyramid level. Image_size : 224 x 224
        # for i, anchor in enumerate(anchors):
        #     print(f'P{i + 2} anchor shape : {anchor.shape}')
        # P2 anchor shape : torch.Size([9408, 4]), 56 x 56 x 3
        # P3 anchor shape : torch.Size([2352, 4]), 28 x 28 x 3
        # P4 anchor shape : torch.Size([588, 4]), 14 x 14 x 3
        # P5 anchor shape : torch.Size([147, 4]), 7 x 7 x 3
        # P6 anchor shape : torch.Size([48, 4]), 4 x 4 x 3

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):

        key = str(grid_sizes) + str(strides)

        if key in self._cache:
            return self._cache[key]

        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors

        return anchors

    def forward(self, image_list, feature_maps):

        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        ## gird_sizes : about feature map P_n / image_size : 224 x 224
        # pprint(grid_sizes)
        # [
        #     torch.Size((56, 56)),
        #     torch.Size((28, 28)),
        #     torch.Size((14, 14)),
        #     torch.Size((7, 7)),
        #     torch.Size((4, 4))
        # ]

        ## strides : about feature map P_n / image_size : 224 x 224
        # pprint(strides)
        # [
        #     [tensor(4), tensor(4)],
        #     [tensor(8), tensor(8)],
        #     [tensor(16), tensor(16)],
        #     [tensor(32), tensor(32)],
        #     [tensor(56), tensor(56)]
        # ]

        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = []

        # print(image_list.image_sizes)
        for i in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)

        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        self._cache.clear()

        return anchors


if __name__ == "__main__":
    import os
    import sys
    add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(add_dir)
    add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
    sys.path.append(add_dir)
    add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
    sys.path.append(add_dir)

    from src.modules.utils.det_utils import ImageList

    im_size = 256
    image1 = torch.randint(256, (1, 3, im_size, im_size))
    image2 = torch.randint(256, (1, 3, im_size, im_size))

    image = ImageList(torch.cat([image1, image2], dim=0), [image1.shape, image2.shape])

    ratios = [4, 8, 16, 32, 56]
    feat_sizes = [im_size//ratio for ratio in ratios]
    feature = list(torch.randn((1, 256, feat_size, feat_size)) for feat_size in feat_sizes)

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    outputs = anchor_generator(image, feature)
    print(outputs)
    print(len(outputs))

    for output in outputs:
        print(output.shape)
