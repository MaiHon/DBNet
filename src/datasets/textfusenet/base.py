import os
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)


import cv2
import copy
import numpy as np
from abc import *
from shapely.geometry import Polygon
from torch.utils.data import Dataset


import torch
from src.datasets.processing import TargetGenerator
from src.datasets.utils import concat_bboxes, poly2boxes, find_validate_gts


base_label = {'word': 1}
num_labels = {
    chr(i): i + 2 - 48 for i in range(48, 48+11)
}

pos = len(num_labels)
lower_labels = {
    chr(i): pos + i - 64  for i in range(65, 65+26)
}

pos = len(lower_labels) + len(num_labels)
upper_labels = {
    chr(i): pos + i - 96 for i in range(97, 97+26)
}

cls2label = {**base_label, **num_labels, **lower_labels, **upper_labels}
label2cls = {v:k for k, v in cls2label.items()}


class TextFuseBaseDS(Dataset):
    def __init__(self, data_path, ignore_tags, filter_keys, pseudo_labels_path=None, transform=None, char_masks=True, return_text=False, **kwargs):
        self.transform = transform
        self.box_type = 'pascal_voc'
        self.ignore_tags = ignore_tags
        self.filter_keys = filter_keys
        self.pseudo_labels_path = pseudo_labels_path
        self.char_masks = char_masks
        self.return_text = return_text

        self.cls2label = cls2label
        self.label2cls = label2cls
        self.target_gen = TargetGenerator(cls2label)
        self.data_list = self.prepare_data(data_path, pseudo_path=pseudo_labels_path)



    @abstractmethod
    def prepare_data(self, data_path, pseudo_path=None):
        raise NotImplementedError


    def __getitem__(self, index):
        try:
            data = copy.deepcopy(self.data_list[index])
            img_p = data['img_fp']
            if 'gif' in img_p:
                gif = cv2.VideoCapture(img_p)
                _, im = gif.read()
            else:
                im = cv2.imread(img_p, 1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


            data['img'] = im
            data['shape'] = im.shape[:2]

            num_words, _, merged_bboxes = concat_bboxes(data.get('wordBBs', None),
                                                        data.get('charBBs', None))
            data['boxes'] = merged_bboxes
            data['num_words'] = num_words


            target = {}
            if self.transform:
                transformed = self.transform(image=data['img'],
                                             bboxes=data['boxes'])

                image = transformed['image']
                valid_boxes, word_or_char, valid_ignores, valid_texts = find_validate_gts(image,
                                                                           transformed['bboxes'].astype(np.float32),
                                                                           data['ignore_tags'], data['num_words'], data['texts'])

                if sum(word_or_char) == 0:
                    return self.__getitem__(np.random.randint(self.__len__()))


                im_shape = image.shape[-2:] if isinstance(image, torch.Tensor) else image.shape[:2]
                tmp_target = {
                    'img': image,
                    'shape': im_shape,
                    'boxes': valid_boxes,
                    'is_word': word_or_char,
                    'texts': valid_texts,
                    'ignore_tags': valid_ignores
                }

                tmp_target = self.target_gen(tmp_target, self.char_masks)
                target['masks'] = tmp_target['masks']
                target['labels'] = tmp_target['labels']
                target['boxes'] = tmp_target['boxes']
                if self.return_text:
                    target['texts'] = valid_texts
                    target['image_fp'] = [data['image_fp']]
            else:
                image = data['img']
                data = self.target_gen(data, self.char_masks, without_tfms=True)
                target['masks'] = data['masks']
                target['boxes'] = data['boxes']
                target['labels'] = data['labels']


            if len(target['boxes']) == 0 or len(target['masks']) == 0 or len(target['labels']) == 0:
                target['img_indx'] = torch.as_tensor(data['img_indx'], dtype=torch.int32)
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target['polygons'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
                target['boxes'] = torch.as_tensor(poly2boxes(target['boxes'], type=self.box_type), dtype=torch.float32)
                target["masks"] = torch.zeros(
                    0, image.shape[0], image.shape[1], dtype=torch.uint8
                )
                target["labels"] = torch.zeros(0, dtype=torch.int64)
                target['shape'] = torch.as_tensor(data['shape'], dtype=torch.int64)
                return image, target

            target['img_indx'] = torch.as_tensor(data['img_indx'], dtype=torch.int64)
            target['polygons'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
            target['boxes'] = torch.as_tensor(poly2boxes(target['boxes'], type=self.box_type), dtype=torch.float32)
            target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
            target['shape'] = torch.as_tensor(data['shape'], dtype=torch.int64)

            return image, target
        except Exception as e:
            return self.__getitem__(np.random.randint(self.__len__()))


    def __len__(self):
        return len(self.data_list)