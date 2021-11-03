# basically from
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/iaa_augment.py

import cv2
import math
import numpy as np

import imgaug
import imgaug.augmenters as iaa
import omegaconf


class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list) or isinstance(args, omegaconf.listconfig.ListConfig):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(*[self.to_tuple_if_list(a) for a in list(args[1:])])
        elif isinstance(args, dict) or isinstance(args, omegaconf.dictconfig.DictConfig):
            cls = getattr(iaa, args['type'])
            return cls(**{k: self.to_tuple_if_list(v) for k, v in args['args'].items()})
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list) or isinstance(obj, omegaconf.listconfig.ListConfig):
            return tuple(obj)
        elif isinstance(obj, omegaconf.dictconfig.DictConfig):
            return dict(obj)
        return obj


class IaaAugment():
    def __init__(self, augmenter_args, only_resize=False, keep_ratio=False):
        self.augmenter_args = augmenter_args
        self.only_resize = only_resize
        self.keep_ratio = keep_ratio
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)


    def __call__(self, data):
        image = data['img']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data['img'] = self.resize_image(image)
            else:
                data['img'] = aug.augment_image(image)
                data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['text_polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['text_polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape['height']
        width = resize_shape['width']
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image