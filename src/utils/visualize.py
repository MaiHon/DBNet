import cv2
import random
import numpy as np
import pyclipper
from shapely.geometry import Polygon

import torch
import torchvision.utils as vutils


def denormalize(batch, mean, std):
    # denorm original img
    cp_img = batch['img'].clone()

    cp_img[:, 0, :, :] = cp_img[:, 0, :, :] * std[0] + mean[0]
    cp_img[:, 1, :, :] = cp_img[:, 1, :, :] * std[1] + mean[1]
    cp_img[:, 2, :, :] = cp_img[:, 2, :, :] * std[2] + mean[2]

    return cp_img


def visuailze_imgs(batch, preds):
    # denorm original img
    b_size = preds.size(0)

    if 'prob_map' in batch:
        probabilty_labels = batch['prob_map'].clone()
        threshold_labels = batch['thresh_map'].clone()
        probabilty_masks = batch['prob_mask'].clone()
        threshold_masks = batch['thresh_mask'].clone()

        show_label = torch.cat([probabilty_labels, threshold_labels, probabilty_masks, threshold_masks])
        show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=b_size, normalize=False, padding=20, pad_value=1)
    else:
        show_label = None


    show_pred = []
    for i in range(preds.size(1)):
        show_pred.append(preds[:, i, :, :])
    show_pred.append((preds[:, 0, :, :] > 0.3))
    show_pred = torch.cat(show_pred)
    show_pred = vutils.make_grid(show_pred.unsqueeze(1), nrow=b_size, normalize=False, padding=20, pad_value=1)
    return show_label, show_pred


class Seg2BoxOrPoly():
    def __init__(self, thresh=0.2, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio


    def __call__(self, pred, batch, is_output_polygon=False, by_seg=False):
        if len(pred.shape) == 4:
            pred = pred[:, 0, :, :].clone()
        else:
            pred = pred.clone()

        # binarize
        segmentation = self.binarize(pred)

        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.size(0)):
            if by_seg: height, width = segmentation.size(1), segmentation.size(2)
            else:
                height, width = batch['shape'][batch_index]
            if is_output_polygon:
                boxes, scores = self.get_polygons(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.get_boxes(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch


    def binarize(self, pred):
        return pred > self.thresh


    def box_scoring(self, segment, box):
        h, w = segment.shape[:2]
        box_ = box.copy()

        xmin = np.clip(np.floor(box_[:, 0].min()).astype(np.int), 0, w-1)
        xmax = np.clip(np.ceil(box_[:, 0].max()).astype(np.int), 0, w-1)
        ymin = np.clip(np.floor(box_[:, 1].min()).astype(np.int), 0, h-1)
        ymax = np.clip(np.ceil(box_[:, 1].max()).astype(np.int), 0, h-1)

        mask = np.zeros((ymax-ymin+1, xmax-xmin+1), dtype=np.uint8)
        box_[:, 0] = box_[:, 0] - xmin
        box_[:, 1] = box_[:, 1] - ymin
        cv2.fillPoly(mask, box_.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(segment[ymin:ymax+1, xmin:xmax+1], mask)[0]


    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded


    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])


    def draw_polygons(self, image, polygons, gt_polygons=None, ignore_tags=None):
        if isinstance(image, torch.Tensor):
            cp_img = image.clone()
            cp_img = cp_img.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        else:
            cp_img = image.copy()


        if isinstance(polygons, dict):
            polygons = np.array([v for k, v in polygons.items()])

        b_size = cp_img.shape[0]
        draw_imgs = []
        for idx, polygon in enumerate(polygons):
            tar_img = cp_img[idx]
            tar_img = cv2.cvtColor(tar_img, cv2.COLOR_RGB2BGR)

            for i in range(len(polygon)):
                poly = np.array(polygon[i]).reshape(-1, 2).astype(np.int32)
                color = (0, 255, 0) # green for prediction
                tar_img = cv2.polylines(tar_img, [poly], True, color, 2)
            tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
            cp_img[idx] = tar_img
            if gt_polygons is None:
                tar_img = torch.from_numpy(tar_img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
                draw_imgs.append(tar_img)

        if gt_polygons is not None:
            for idx, polygon in enumerate(gt_polygons):
                tar_img = cp_img[idx]
                tar_img = cv2.cvtColor(tar_img, cv2.COLOR_RGB2BGR)

                for i in range(len(polygon)):
                    poly = np.array(polygon[i]).reshape(-1, 2).astype(np.int32)
                    if ignore_tags[idx] is not None:
                        ignore = ignore_tags[idx][i]
                        if ignore:
                            color = (255, 0, 0)  # depict ignorable polygons in blue
                        else:
                            color = (0, 0, 255)  # depict polygons in red
                    tar_img = cv2.polylines(tar_img, [poly], True, color, 2)
                tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
                tar_img = torch.from_numpy(tar_img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
                draw_imgs.append(tar_img)

        for i in range(len(draw_imgs), b_size):
            tar_img = cp_img[i]
            tar_img = torch.from_numpy(tar_img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            draw_imgs.append(tar_img)

        draw_imgs = torch.cat(draw_imgs, dim=0)
        draw_imgs = vutils.make_grid(draw_imgs, nrow=b_size, normalize=False, padding=20, pad_value=1)
        return draw_imgs


    def get_polygons(self, pred, segment, dest_w, dest_h):
        segment = segment.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        segment_h, segment_w = segment.shape[:2]
        boxes = []
        scores = []

        contours, _ = cv2.findContours((segment*255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE -> 근사화된 영역 찾기
        for contour in contours[:self.max_candidates]:
            epsilon = .005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))

            if points.shape[0] < 4:
                continue

            score = self.box_scoring(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points)
                if len(box) > 1:
                    continue
            else:
                continue

            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_w, int):
                dest_w = dest_w.item()
                dest_h = dest_h.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / segment_w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / segment_h * dest_h), 0, dest_h)
            boxes.append(box)
            scores.append(score)
        return boxes, scores


    def get_boxes(self, pred, _segment, dest_w, dest_h):
        '''
            get bbox coords from binarized segment
        '''

        segment = _segment.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        segment_h, segment_w = segment.shape
        contours, _ = cv2.findContours((segment*255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            points = np.array(points)
            # score = self.box_scoring(pred, points.reshape(-1, 2))
            score = self.box_scoring(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            if not isinstance(dest_w, int):
                dest_w = dest_w.item()
                dest_h = dest_h.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / segment_w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / segment_h * dest_h), 0, dest_h)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores