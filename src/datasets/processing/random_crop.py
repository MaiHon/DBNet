import cv2
import numpy as np
from shapely.geometry import Polygon


class RandomCrop:
    def __init__(self,
                crop_size=(640,640), max_tries=50, keep_ratio=True,
                min_crop_side_ratio=0.1, require_original_image=False):
        self.crop_size = crop_size
        self.max_tries = max_tries
        self.keep_ratio = keep_ratio
        self.min_crop_side_ratio = min_crop_side_ratio
        self.require_original_image = require_original_image


    def __call__(self, data):
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']
        texts = data['texts']

        valid_polys = [text_polys[i] for i, tag in enumerate(ignore_tags) if not tag]
        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, valid_polys)


        # crop_size
        scale_w = self.crop_size[0] / crop_w
        scale_h = self.crop_size[1] / crop_h
        scale = min(scale_w, scale_h)

        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            if len(im.shape) == 3:
                img = np.zeros((self.crop_size[1], self.crop_size[0], im.shape[2]), im.dtype)
            else:
                img = np.zeros((self.crop_size[1], self.crop_size[0]), im.dtype)
            img[:h, :w] = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))

        else:
            img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], tuple(self.crop_size))

        # check valid polys
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if (not self.is_poly_outside_rect(poly, 0, 0, w, h)):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)

        data['img'] = img
        data['text_polys'] = np.float32(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        data['texts'] = texts_crop
        return data


    def crop_area(self, image, valid_polys):
        """
            main crop algorithm
            
        """
        h, w = image.shape[:2]
        w_axis, h_axis = self.get_axis(image, valid_polys)
        if len(h_axis) == 0 or len(w_axis) == 0:
            # 모든 영역에 박스가 존재한다면 이미지 전체를 crop 영역으로 지정
            return 0, 0, w, h


        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        min_crop_w = self.min_crop_side_ratio * w
        min_crop_h = self.min_crop_side_ratio * h
        
        for i in range(self.max_tries):
            # 추출할 영역이 있다면 random crop
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            # 크롭 후보 영역이 너무 작다면 패스
            if xmax - xmin < min_crop_w or ymax - ymin < min_crop_h:
                continue
            
            # valid한 polygon들 중에서도, 크롭한 영역 (xmin, ymin, xmax, ymax)
            # 안에 존재하는 polygon이 존재하는지 확인하기
            num_poly_in_rect = 0
            for poly in valid_polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        # max_tries 만큼 시도한 후에
        # 하나의 크롭도 성공하지 못한다면 이미지 전체를 크롭영역으로 넘겨줌
        return 0, 0, w, h


    def get_axis(self, image, polys):
        """
           각각의 축마다 박스가 하나도 존재하지 않는 축들을 뽑아냄
        """
        h, w = image.shape[:2]
        h_axis = np.zeros((h), dtype=np.int32)
        w_axis = np.zeros((w), dtype=np.int32)

        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)

            xmin = np.min(poly[:, 0])
            xmax = np.max(poly[:, 0])
            ymin = np.min(poly[:, 1])
            ymax = np.max(poly[:, 1])

            w_axis[xmin:xmax] = 1
            h_axis[ymin:ymax] = 1

        h_axis = np.where(h_axis==0)[0]
        w_axis = np.where(w_axis==0)[0]
        return w_axis, h_axis


    def split_regions(self, axis):
        """
           박스가 존재 하지 않는 지점들을 닮고 있는 axis를 통해서
           경계선을 찾음
        """
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i-1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions


    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax


    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)

        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax


    def is_poly_outside_rect(self, poly, x, y, w, h):
        crop_polygon = Polygon([(x, y), (x+w-1, y), (x+w-1, y+h-1), (x, y+h-1)])
        cand_polygon = Polygon(poly.copy())
        if cand_polygon.area < 10:
            return True
        
        intersect = crop_polygon.intersection(cand_polygon)
        if intersect.area < 10:
            return True

        return False
    
    
if __name__ == "__main__":
    print("Random Crop Test")
    
    cropper = RandomCrop()
    data = {
        'img': np.zeros((900, 500, 3)),
        'text_polys': [
            np.array([[34, 56], [89, 56], [89, 79], [34, 79]]),
            np.array([[128, 324], [128+325, 324], [128+325, 324+56], [128, 324+56]])
        ],
        'ignore_tags': [False, False],
        'texts': ['Test', 'Test']
    }
    
    ori_im = data['img'].copy()
    ori_polys = data['text_polys'].copy()

    cropped_data = cropper(data)
    cropped_im = data['img']
    for poly in data['text_polys']:
        ptr1 = (int(poly[:, 0].min()), int(poly[:, 1].min()))
        ptr2 = (int(poly[:, 0].max()), int(poly[:, 1].max()))
        cv2.rectangle(cropped_im, ptr1, ptr2, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        
    for poly in ori_polys:
        ptr1 = (poly[:, 0].min(), poly[:, 1].min())
        ptr2 = (poly[:, 0].max(), poly[:, 1].max())
        cv2.rectangle(ori_im, ptr1, ptr2, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    
    
    cv2.imshow('cropped_im', cropped_im)
    cv2.imshow('ori_im', ori_im)
    cv2.waitKey()
    cv2.destroyAllWindows()