import cv2
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

import pyclipper
from shapely.geometry import Polygon


class ThresholdMapGenerator:
    """
        Generating threshold map for DBNet
    """
    def __init__(self, shrink_ratio=0.4, min_thresh=0.3, max_thresh=0.7, same_offset=True):
        self.shrink_ratio = shrink_ratio
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.same_offset = same_offset


    def __call__(self, data):
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        canvas = np.zeros((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)

        for i in range(len(text_polys)):
            if ignore_tags[i]: continue

            self.draw_threshold_map(text_polys[i], canvas, mask=mask)
        canvas = canvas * (self.max_thresh - self.min_thresh) + self.min_thresh

        data['thresh_map'] = canvas
        data['thresh_mask'] = mask
        return data


    def draw_threshold_map(self, polygon, canvas, mask):
        if self.same_offset:
            polygon = np.array(polygon)
            polygon_shape = Polygon(polygon)
            if polygon_shape.area <= 0:
                return

            distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
            subject = [tuple(l) for l in polygon]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

            dilated_polygon = np.array(padding.Execute(distance)[0])
            cv2.fillPoly(mask, [dilated_polygon.astype(np.int32)], 1.0)
        else:
            raise NotImplemented

        xmin = dilated_polygon[:, 0].min()
        xmax = dilated_polygon[:, 0].max()
        ymin = dilated_polygon[:, 1].min()
        ymax = dilated_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        x = np.linspace(0, width-1, num=width)
        y = np.linspace(0, height-1, num=height)
        xs, ys = np.meshgrid(x, y)

        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.calc_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)


        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
                1 - distance_map[
                    ymin_valid - ymin:ymax_valid - ymax + height,
                    xmin_valid - xmin:xmax_valid - xmax + width],
                canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


    def calc_distance(self, xs, ys, point_i, point_j):
        height, width = xs.shape[:2]
        square_dist_1 = np.square(xs - point_i[0]) + np.square(ys - point_i[1])
        square_dist_2 = np.square(xs - point_j[0]) + np.square(ys - point_j[1])
        square_dist   = np.square(point_i[0] - point_j[0]) + np.square(point_i[1] - point_j[1])


        # 코사인 제2법칙
        cosin = (square_dist - square_dist_1 - square_dist_2) / (2 * np.sqrt(square_dist_1 * square_dist_2))
        sin = 1 - np.square(cosin)
        sin = np.nan_to_num(sin)


        # 삼각형 넓이 공식 a*b*sin(theta) == result * square_dist
        #TODO: 왜 cosine값이 0보다 작은, 둔각의 경우는 다르게 고려하는지?
        result = np.sqrt(square_dist_1 * square_dist_2 * sin / square_dist)
        result[cosin < 0] = np.sqrt(np.fmin(square_dist_1, square_dist_2))[cosin < 0]
        return result