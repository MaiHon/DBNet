import cv2
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import distance as dist


def validate_polygons(polygon, h, w):
    try:
        valid_polygon = Polygon(polygon.copy())
        image_polygon = Polygon([(0,0), (w-1, 0), (w-1, h-1), (0, h-1)])
        intersect = image_polygon.intersection(valid_polygon)
        if intersect.area < 1:
            return False, None

        return_polygon = polygon.copy()
        return_polygon[:, 0] = np.clip(return_polygon[:, 0], 0, w-1)
        return_polygon[:, 1] = np.clip(return_polygon[:, 1], 0, h-1)

        area = cv2.contourArea(return_polygon)
        if area < 1:
            return False, None
        if area > 0:
            return True, return_polygon
    except Exception as e:
        return False, None



def find_validate_gts(image, boxes, ignores, num_words, texts):
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        h, w = image.shape[-2:]

    valid_boxes = []
    word_or_char = []
    valid_texts = []
    valid_ignores = []

    for idx, (box, ignore, text) in enumerate(zip(boxes, ignores, texts)):
        ret, polygon = validate_polygons(box, h, w)
        if ret:
            valid_boxes.append(polygon)
            word_or_char.append(idx < num_words)
            valid_texts.append(text)
            valid_ignores.append(ignore)

    return np.array(valid_boxes), word_or_char, valid_ignores, valid_texts


def concat_bboxes(word_bboxes=None, by_word_char_bboxes=None):
        """
        Returns:
            num_word_bboxes (int)
            num_char_bboxes (int)
            merged_bboxes (ndarray): (N, 4, 2) shaped ndarray.
        """
        if word_bboxes is None:
            bboxes_word = np.ndarray((0, 4, 2), dtype=np.float32)
        else:
            bboxes_word = np.array(word_bboxes).reshape(-1, 4, 2)
        if by_word_char_bboxes is None:
            bboxes_char = np.ndarray((0, 4, 2), dtype=np.float32)
        else:
            for idx, char_bboxes in enumerate(by_word_char_bboxes):
                if len(char_bboxes) == 0:
                    by_word_char_bboxes[idx] = np.ndarray((0, 4, 2), dtype=np.float32)
            bboxes_char = np.concatenate(by_word_char_bboxes, axis=0)
        merged_bboxes = np.concatenate((bboxes_word, bboxes_char), axis=0)

        return len(bboxes_word), len(bboxes_char), merged_bboxes


def poly2boxes(polys, type='pascal_voc'):
    boxes = []

    if type == 'pascal_voc':
        for poly in polys:
            x1, y1 = np.min(poly[:, 0]), np.min(poly[:, 1])
            x2, y2 = np.max(poly[:, 0]), np.max(poly[:, 1])

            boxes.append([x1, y1, x2, y2])
    else:
        raise NotImplemented

    return boxes


def charBB2wordBB(charBBs, merge_method='rect'):
    if merge_method != 'rect':
        raise NotImplementedError

    word_bboxes = []
    for bboxes in map(np.array, charBBs):
        if bboxes.ndim == 2:  # [N, 4]
            word_bboxes.append(
                [bboxes[:, 0].min(), bboxes[:, 1].min, bboxes[:, 2].max(), bboxes[:, 3].max()])
        else:  # [N, 4, 2]
            xmin, ymin = bboxes[:, :, 0].min(), bboxes[:, :, 1].min()
            xmax, ymax = bboxes[:, :, 0].max(), bboxes[:, :, 1].max()
            word_bboxes.append(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return word_bboxes


def is_valid_polygon(poly):
    candi_poly = Polygon(poly)
    return candi_poly.is_valid


def polygon_validity_check(text_polys, transcripts):
    valid_polys, valid_transcripts = [], []

    for poly, transcript in zip(text_polys, transcripts):

        if is_valid_polygon(poly):
            valid_polys.append(poly)
            valid_transcripts.append(transcript)
        else:
            poly = order_points(poly)
            if is_valid_polygon(poly):
                valid_polys.append(poly)
                valid_transcripts.append(transcript)

    return np.array(valid_polys, dtype=text_polys.dtype), valid_transcripts


# from https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")