import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import distance as dist


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