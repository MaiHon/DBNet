import cv2
import numpy as np

import pyclipper
from shapely.geometry import Polygon




class ProbabiltyMapGenerator:
    """
        Generating probabilty map for DBNet
    """

    def __init__(self, shrink_ratio=0.4, min_text_size=8, same_offset=True):
        self.shrink_ratio = shrink_ratio
        self.min_text_size = min_text_size
        self.same_offset = same_offset


    def __call__(self, data):
        """
            :param data: {'img', 'text_polys', 'texts', 'ignore_tags'}
        """
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)

        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])


            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, : ], 0)
                ignore_tags[i] = True
            else:
                shrinked_polygon = self.shrink_polygon(polygon, self.shrink_ratio, self.same_offset)
                if shrinked_polygon.size == 0:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue

                cv2.fillPoly(gt, [shrinked_polygon.astype(np.int32)], 1)

        data['prob_map'] = gt
        data['prob_mask'] = mask
        return data


    def shrink_polygon(self, polygon, shrink_ratio, same_offset=True):
        if same_offset:
            polygon_shape = Polygon(polygon)
            distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length

            subject = [tuple(l) for l in polygon]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_polygon = padding.Execute(-distance)

            if shrinked_polygon == []:
                shrinked_polygon = np.array(shrinked_polygon)
            else:
                shrinked_polygon = np.array(shrinked_polygon[0]).reshape(-1, 2)

            return shrinked_polygon
        #TODO: giving different offset w.r.t hight of each ponits
        else:
            raise NotImplemented


    def validate_polygons(self, polygons, ignore_tags, h, w):
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)

        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w-1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h-1)

        for i in range(len(polygons)):
            area = cv2.contourArea(polygons[i])
            if abs(area) < 10:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags