import cv2
import numpy as np
import numpy.random as npr
from shapely.geometry import Polygon


class GeoTransform:
    def __init__(
        self,
        rotate_anchors=None, rotate_range=None,
        crop_aspect_ratio=None, crop_size=1.0, crop_size_by='longest', hflip=False, vflip=False,
        cond_random_translate=False, min_image_overlap=0.5, min_bbox_overlap=0.90, min_bbox_count=0,
        resize_to=None, keep_aspect_ratio=False, resize_based_on='longest', max_random_trials=100
    ):

        if rotate_anchors is None:
            self.rotate_anchors = None
        elif type(rotate_anchors) in [int, float]:
            self.rotate_anchors = [rotate_anchors]
        elif len(rotate_anchors) == 2:
            self.rotate_anchors = list(rotate_anchors)

        if rotate_range is None:
            self.rotate_range = None
        elif type(rotate_range) in [int, float]:
            assert rotate_range >= 0
            self.rotate_range = (-rotate_range, rotate_range)
        elif len(rotate_range) == 2:
            assert rotate_range[0] <= rotate_range[1]
            self.rotate_range = tuple(rotate_range)
        else:
            raise TypeError

        if crop_aspect_ratio is None:
            self.crop_aspect_ratio = None
        elif type(crop_aspect_ratio) in [int, float]:
            self.crop_aspect_ratio = float(crop_aspect_ratio)
        elif len(crop_aspect_ratio) == 2:
            self.crop_aspect_ratio = tuple(crop_aspect_ratio)
        else:
            raise TypeError

        if type(crop_size) in [int, float]:
            self.crop_size = crop_size
        elif len(crop_size) == 2:
            assert type(crop_size[0]) == type(crop_size[1])
            self.crop_size = tuple(crop_size)
        else:
            raise TypeError

        assert crop_size_by in ['width', 'height', 'longest']
        self.crop_size_by = crop_size_by

        self.hflip, self.vflip = hflip, vflip

        self.cond_random_translate = cond_random_translate

        self.min_image_overlap = min_image_overlap
        self.min_bbox_overlap = min_bbox_overlap
        self.min_bbox_count = min_bbox_count

        self.max_random_trials = max_random_trials

        if resize_to is None:
            self.resize_to = resize_to
        elif type(resize_to) in [int, float]:
            if not keep_aspect_ratio:
                self.resize_to = (resize_to, resize_to)
            else:
                self.resize_to = resize_to
        elif len(resize_to) == 2:
            assert not keep_aspect_ratio
            assert type(resize_to[0]) == type(resize_to[1])
            self.resize_to = tuple(resize_to)
        assert resize_based_on in ['width', 'height', 'longest']
        self.keep_aspect_ratio, self.resize_based_on = keep_aspect_ratio, resize_based_on

    def __call__(self, image, bboxes=[], masks=[]):
        return self.crop_rotate_resize(image, bboxes=bboxes, masks=masks)

    def _get_theta(self):
        if self.rotate_anchors is None:
            theta = 0
        else:
            theta = npr.choice(self.rotate_anchors)
        if self.rotate_range is not None:
            theta += npr.uniform(*self.rotate_range)

        return theta

    def _get_patch_size(self, ih, iw):
        if (self.crop_aspect_ratio is None and isinstance(self.crop_size, float) and
            self.crop_size == 1.0):
            return ih, iw

        if self.crop_aspect_ratio is None:
            aspect_ratio = iw / ih
        elif isinstance(self.crop_aspect_ratio, float):
            aspect_ratio = self.crop_aspect_ratio
        else:
            aspect_ratio = np.exp(npr.uniform(*np.log(self.crop_aspect_ratio)))

        if isinstance(self.crop_size, tuple):
            if isinstance(self.crop_size[0], int):
                crop_size = npr.randint(self.crop_size[0], self.crop_size[1])
            elif self.crop_size[0]:
                crop_size = np.exp(npr.uniform(*np.log(self.crop_size)))
        else:
            crop_size = self.crop_size

        if self.crop_size_by == 'longest' and iw >= ih or self.crop_size_by == 'width':
            if isinstance(crop_size, int):
                pw = crop_size
                ph = int(pw / aspect_ratio)
            else:
                pw = int(iw * crop_size)
                ph = int(iw * crop_size / aspect_ratio)
        else:
            if isinstance(crop_size, int):
                ph = crop_size
                pw = int(ph * aspect_ratio)
            else:
                ph = int(ih * crop_size)
                pw = int(ih * crop_size * aspect_ratio)

        return ph, pw

    def _get_patch_quad(self, theta, ph, pw):
        cos, sin = np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)
        hpx, hpy = 0.5 * pw, 0.5 * ph  # half patch size
        quad = np.array([[-hpx, -hpy], [hpx, -hpy], [hpx, hpy], [-hpx, hpy]], dtype=np.float32)
        rotation_m = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)
        quad = np.matmul(quad, rotation_m)  # patch quadrilateral in relative coords

        return quad

    def _get_located_patch_quad(self, ih, iw, patch_quad_rel, bboxes=[]):
        if self.min_image_overlap is not None:
            image_poly = Polygon([[0, 0], [iw, 0], [iw, ih], [0, ih]])
            center_patch_poly = Polygon(
                np.array([0.5 * ih, 0.5 * iw], dtype=np.float32) + patch_quad_rel)
            max_available_overlap = (image_poly.intersection(center_patch_poly).area /
                                     center_patch_poly.area)
            min_image_overlap = min(self.min_image_overlap, max_available_overlap)
        else:
            min_image_overlap = None

        if self.min_bbox_count > 0:
            min_bbox_count = min(self.min_bbox_count, len(bboxes))
        else:
            min_bbox_count = 0

        cx_margin, cy_margin = np.sort(patch_quad_rel[:, 0])[2], np.sort(patch_quad_rel[:, 1])[2]

        found_randomly = False
        for _ in range(self.max_random_trials):
            cx, cy = npr.uniform(cx_margin, iw - cx_margin), npr.uniform(cy_margin, ih - cy_margin)
            patch_quad = np.array([cx, cy], dtype=np.float32) + patch_quad_rel
            patch_poly = Polygon(patch_quad)

            image_overlap = patch_poly.intersection(image_poly).area / patch_poly.area
            if min_image_overlap is not None and image_overlap < min_image_overlap:
                continue

            if min_bbox_count > 0:
                bbox_polys = [poly for poly in [Polygon(bbox) for bbox in bboxes] if poly.area > 0]
                bbox_overlaps = np.array(
                    [poly.convex_hull.intersection(patch_poly).area / poly.area for poly in bbox_polys])
                if np.count_nonzero(bbox_overlaps >= self.min_bbox_overlap) < min_bbox_count:
                    continue
                if np.logical_and(bbox_overlaps > 0, bbox_overlaps < self.min_bbox_overlap).any():
                    continue

            found_randomly = True
            break

        return patch_quad if found_randomly else None

    def crop_rotate_resize(self, image, bboxes=[], masks=[]):
        ih, iw = image.shape[:2]  # image height and width

        theta = self._get_theta()
        ph, pw = self._get_patch_size(ih, iw)

        patch_quad_rel = self._get_patch_quad(theta, ph, pw)

        if not self.cond_random_translate:
            cx, cy = 0.5 * iw, 0.5 * ih
            patch_quad = np.array([cx, cy], dtype=np.float32) + patch_quad_rel
        else:
            patch_quad = self._get_located_patch_quad(ih, iw, patch_quad_rel, bboxes=bboxes)

        vflip, hflip = self.vflip and npr.randint(2) > 0, self.hflip and npr.randint(2) > 0

        if self.resize_to is None:
            oh, ow = ih, iw
        elif self.keep_aspect_ratio:  # `resize_to`: Union[int, float]
            if self.resize_based_on == 'height' or self.resize_based_on == 'longest' and ih >= iw:
                oh = ih * self.resize_to if isinstance(self.resize_to, float) else self.resize_to
                ow = int(oh * iw / ih)
            else:
                ow = iw * self.resize_to if isinstance(self.resize_to, float) else self.resize_to
                oh = int(ow * ih / iw)
        elif isinstance(self.resize_to[0], float):  # `resize_to`: tuple[float, float]
            oh, ow = ih * self.resize_to[0], iw * self.resize_to[1]
        else:  # `resize_to`: tuple[int, int]
            oh, ow = self.resize_to

        if theta == 0 and (ph, pw) == (ih, iw) and (oh, ow) == (ih, iw) and not (hflip or vflip):
            patch = image
            masks_tr = masks
            M = None
        else:
            dst = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], dtype=np.float32)
            if patch_quad is not None:
                src = patch_quad
            else:
                if ow / oh >= iw / ih:
                    pad = int(ow * ih / oh) - iw
                    off = npr.randint(pad + 1)  # offset
                    src = np.array(
                        [[-off, 0], [iw + pad - off, 0], [iw + pad - off, ih], [-off, ih]],
                        dtype=np.float32)
                else:
                    pad = int(oh * iw / ow) - ih
                    off = npr.randint(pad + 1)  # offset
                    src = np.array(
                        [[0, -off], [iw, -off], [iw, ih + pad - off], [0, ih + pad - off]],
                        dtype=np.float32)

            if hflip:
                src = src[[1, 0, 3, 2]]
            if vflip:
                src = src[[3, 2, 1, 0]]

            M = cv2.getPerspectiveTransform(src, dst)
            patch = cv2.warpPerspective(image, M, dsize=(ow, oh))
            masks_tr = [cv2.warpPerspective(mask, M, dsize=(ow, oh)) for mask in masks]

            if len(bboxes) > 0:
                bboxes = cv2.perspectiveTransform(
                    np.reshape(bboxes, (1, -1, 2)).astype(np.float32), M).reshape(-1, 4, 2).astype(
                        np.int32)

        found_randomly = self.cond_random_translate and patch_quad is not None

        return dict(image=patch, bboxes=bboxes, masks=masks_tr, found_randomly=found_randomly,
                    matrix=M)
