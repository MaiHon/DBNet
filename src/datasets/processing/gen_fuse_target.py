import cv2
import torch
import numpy as np


class TargetGenerator:
    """
        Generating Mask data for TextFuseNet
    """

    def __init__(self, tot_labels):
        self.tot_labels = tot_labels


    def __call__(self, data, char_masks=True, without_tfms=False):
        """
            :param data: {'img', 'text_polys', 'texts', 'ignore_tags'}
        """

        if without_tfms:
            image = data['img']
            char_polys = data['boxes']
            num_words  = data['num_words']
            transcriptions = data['texts']
            ignore_tags = data['ignore_tags']

            h, w = image.shape[:2]

            words = char_polys[:num_words]
            chars = char_polys[num_words:]

            pos = 0
            masks = []
            labels = []
            for idx, text in enumerate(transcriptions):
                if not ignore_tags[idx]:
                    word_gt = np.zeros((h, w), dtype=np.float32)
                    word_poly = words[idx]
                    masks.append(cv2.fillPoly(word_gt, [word_poly.astype(np.int32)], 1))
                    labels.append(0)

                    if chars and char_masks:
                        for i in range(len(text)):
                            gt = np.zeros((h, w), dtype=np.float32)
                            labels.append(self.tot_labels[text[i]])
                            char_poly = chars[pos+i]
                            masks.append(cv2.fillPoly(gt, [char_poly.astype(np.int32)], 1))

                    pos += len(text)

            data['masks'] = masks
            data['labels'] = labels
            data['shape'] = [h, w]
        else:
            char_polys = data['boxes']
            is_word = data['is_word']
            texts = data['texts']
            ignore_tags = data['ignore_tags']
            h, w = data['shape']


            masks = []
            labels = []
            for idx, poly in enumerate(char_polys):
                if ignore_tags[idx]: continue

                if is_word[idx]:
                    labels.append(1)
                    word_gt = np.zeros((h, w), dtype=np.float32)
                    mask = cv2.fillPoly(word_gt, [poly.astype(np.int32)], 1)
                elif char_masks:
                    gt = np.zeros((h, w), dtype=np.float32)

                    if texts[idx] in self.tot_labels:
                        labels.append(self.tot_labels[texts[idx]])
                    else:
                        # 특수문자에 해당하는 경우?
                        labels.append(len(self.tot_labels) + 1)
                    mask = cv2.fillPoly(gt, [poly.astype(np.int32)], 1)

                masks.append(torch.tensor(mask, dtype=torch.uint8))

            data['masks'] = torch.stack(masks)
            data['labels'] = labels
            data['shape'] = [h, w]

            assert len(masks) == len(labels) == len(char_polys)
        return data