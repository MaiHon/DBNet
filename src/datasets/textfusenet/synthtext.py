import os
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)
add_dir = os.path.abspath(os.path.curdir)
sys.path.append(add_dir)


import re
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from itertools import chain


import torch
from torch.utils.data import Dataset
from src.datasets.textfusenet import TextFuseBaseDS
from src.datasets.utils import polygon_validity_check, charBB2wordBB
# from src.utils import ComposedTransformation, GeoTransformation



class TextFuseNetSynthTextDS(TextFuseBaseDS):
    def __init__(self, root_p, annots_p, ignore_tags, filter_keys, pseudo_labels_path=None, transform=None, **kwargs):
        self.root_p = root_p
        self.annots_p = annots_p
        self.ignore_tags = ignore_tags
        self.filter_keys = filter_keys

        super().__init__(annots_p, ignore_tags, filter_keys, pseudo_labels_path, transform, **kwargs)


    def prepare_data(self, annots_p, pseudo_path=None):
        print("\nLoading GT...")
        print("It take some time...\n")
        self.annots = sio.loadmat(annots_p)
        self.iter = zip(self.annots['charBB'][0], self.annots['wordBB'][0], self.annots['imnames'][0], self.annots['txt'][0])

        data_list = []
        for idx, (charBBs, wordBBs, in_name, transcriptions) in enumerate(self.iter):
            image_fp = self.root_p / in_name[0]

            transcriptions = [re.split(' \n|\n |\n| ', t.strip()) for t in transcriptions]
            transcriptions = list(chain(*transcriptions))
            transcriptions = [t for t in transcriptions if len(t) > 0]

            by_word_charBBs = []
            charBBs = charBBs.transpose((2, 1, 0))  # [2, 4, N_chars] -> [N_chars, 4, 2]
            cursor = 0
            for word in transcriptions:
                charBB = charBBs[cursor:cursor+len(word)].copy()
                by_word_charBBs.append(charBB)
                cursor += len(word)


            item = {}
            item['img_fp'] = str(image_fp)
            item['img_indx'] = idx
            item['wordBBs'] = charBB2wordBB(by_word_charBBs)
            item['charBBs'] = by_word_charBBs
            item['texts'] = [text for text in transcriptions if text != '']
            item['ignore_tags'] = [x in self.ignore_tags for x in item['texts']]
            item['texts'] += [char for word in item['texts'] for char in word]


            char_ignores = []
            for text, tag in zip(item['texts'], item['ignore_tags']):
                if tag:
                    char_ignores.extend([True] * len(text))
                else:
                    char_ignores.extend([False] * len(text))
            item['ignore_tags'].extend(char_ignores)

            data_list.append(item)
        return data_list


if __name__ == '__main__':
    import os
    import os.path as osp
    import cv2
    import matplotlib.pyplot as plt

    from tqdm import tqdm
    from pathlib import Path
    from skimage import io, color
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from src.datasets.transform import BasicTransform


    root = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir, osp.pardir))
    workspace = Path(root) / 'workspace'
    conf_dir  = workspace / 'configs' / 'experiments'
    conf_name = 'synthtext_textfusenet'
    conf_path = conf_dir / f'{conf_name}.yaml'

    main_cfg = OmegaConf.load(str(conf_path))
    base_cfg = OmegaConf.load(main_cfg['train']['base'])
    main_cfg['train']['dataset'] = base_cfg['dataset']

    ds_name = main_cfg['train']['dataset']['name']
    data_path = Path(root) / main_cfg['train']['root']
    root_path = data_path / main_cfg['train']['dataset']['root_path']
    annot_path = data_path / main_cfg['train']['dataset']['annot_path']

    ignore_tags = main_cfg['train']['dataset']['args']['ignore_tags']
    filter_keys = main_cfg['train']['dataset']['args']['filter_keys']
    tfms_cfg = main_cfg['train']['transform']

    def collate_fn(batch):
        return tuple(zip(*batch))

    tfms_name = tfms_cfg.pop('name')
    tfms = eval(tfms_name)(**tfms_cfg)
    trn_ds = TextFuseNetSynthTextDS(root_path, annot_path, ignore_tags, filter_keys, transform=tfms)
    trn_dl = DataLoader(trn_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    label_names = trn_ds.label2cls


    cnt = 0
    show = 100
    vis = True
    if vis:
        debug = workspace / 'debug' / 'textfusenet' / 'synthtext'
        if not osp.exists(debug):
            os.makedirs(debug)

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(trn_ds.cls2label)+2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for batch in tqdm(trn_dl, total=len(trn_dl)):
        image, target = batch
        b_size = len(image)

        for b_idx in range(b_size):
            image_np = image[b_idx].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            masks = target[b_idx]['masks']
            boxes = target[b_idx]['boxes']
            labels = target[b_idx]['labels']

            assert len(masks) == len(boxes) == len(labels)

            if vis:
                word_char_mask = np.zeros(image_np.shape[:2])
                for i in range(len(masks)):
                    label = labels[i]
                    mask = masks[i]

                    if label == 1:
                        word_char_mask[mask==1] = 1
                    else:
                        word_char_mask[mask==1] = 2

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(coord.item()) for coord in box]
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                    label = labels[i].item()
                    if label in label_names:
                        cv2.putText(image_np, label_names[labels[i].item()], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                thickness=1, lineType=cv2.LINE_AA)
                    else:
                        cv2.putText(image_np, 'special', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                                thickness=1, lineType=cv2.LINE_AA)

                word_char_mask = word_char_mask.astype(np.uint8)
                draw_image = color.label2rgb(word_char_mask, image_np, colors=[(255, 0, 0), (0, 0, 255)], alpha=0.01, bg_label=0, bg_color=None)

                img_save_p = osp.join(debug, f"img_{cnt}.jpg")
                mask_save_p = osp.join(debug, f"draw_{cnt}.jpg")
                cv2.imwrite(mask_save_p, (draw_image*255.).astype(np.uint8))
                cv2.imwrite(img_save_p, image_np)

        cnt += 1
        if cnt == show:
            break