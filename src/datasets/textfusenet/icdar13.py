import os
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)
add_dir = os.path.abspath(os.path.curdir)
sys.path.append(add_dir)

import cv2
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.datasets.textfusenet import TextFuseBaseDS


class TextFuseNetICDAR13DS(TextFuseBaseDS):
    def __init__(self, root_p, annots_p, ignore_tags, filter_keys, pseudo_labels_path=None, transform=None, split='train', char_masks=False, **kwargs):
        self.root_p = root_p
        self.annots_p = annots_p
        self.ignore_tags = ignore_tags
        self.filter_keys = filter_keys
        self.split = split

        super().__init__(annots_p, ignore_tags, filter_keys, pseudo_labels_path, transform, char_masks, **kwargs)


    def prepare_data(self, annots_p, pseudo_path=None):
        print("\nLoading GT...")
        print("It take some time...\n")


        if pseudo_path is not None:
            pseudo_path = Path(pseudo_path)

        annots_path = Path(annots_p)
        annot_names = sorted(os.listdir(str(annots_path)))

        data_list = []
        root_p = Path(self.root_p)
        for idx, annot_name in enumerate(annot_names):
            annot_path = annots_path / annot_name
            annot = self.get_annotation(str(annot_path))

            if self.split == "train":
                img_name = annot_name.split('_')[-1].split('.')[0]
            else:
                img_name = "_".join(annot_name.split('_')[1:])
                img_name = img_name.split('.')[0]

            pseudo_name = img_name + '.json'
            if pseudo_path is not None:
                pseudo_fp   = pseudo_path / pseudo_name
            else:
                pseudo_fp = None

            annot = self.get_pseudo_annotation(str(annot_path), str(pseudo_fp))


            img_name += '.jpg'
            image_fp = root_p / img_name
            if len(annot['wordBBs']) > 0:
                item = {'img_fp': str(image_fp), 'img_name': image_fp.stem, 'img_indx': idx}
                item.update(annot)
                data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(annot_path))
        return data_list


    def get_annotation(self, annot_p):
        bboxes = []
        texts = []
        ignores = []

        with open(str(annot_p), 'r') as f:
            lines = f.readlines()
            for line in lines:
                info = line.split(', ') if self.split == 'valid' else line.split()

                label = info[-1].strip()[1:-1]
                x1, y1, x2, y2 = [int(t) for t in info[:4]]
                bbox = np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.int32).reshape((4, 2))
                if cv2.contourArea(bbox) > 0:
                    bboxes.append(bbox)
                    texts.append(label)
                    ignores.append(label in self.ignore_tags)

        data = {
            'text_polys': np.array(bboxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data


    def get_pseudo_annotation(self, annot_p, pseudo_path=None):
        pseudo = True if pseudo_path is not None and os.path.exists(pseudo_path) else False

        bboxes = []
        char_bboxes = []
        char_texts = []
        char_ignores = []
        texts = []
        ignores = []

        if pseudo:
            with open(pseudo_path, 'r') as f:
                pseudo_data = json.load(f)
        else:
            pseudo_data = None

        with open(str(annot_p), 'r') as f:
            lines = f.readlines()
            for line in lines:
                info = line.split(', ') if self.split == 'valid' else line.split()

                label = info[-1].strip()[1:-1]
                x1, y1, x2, y2 = [int(t) for t in info[:4]]

                bbox = np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.int32).reshape((4, 2))
                if cv2.contourArea(bbox) > 0:
                    bboxes.append(bbox)
                    texts.append(label)
                    ignores.append(label in self.ignore_tags)

                    if len(label) == 1:
                        char_bboxes.append(bbox)
                        char_texts.append(label)
                        char_ignores.append(label in self.ignore_tags)
                    elif pseudo and label in pseudo_data:
                        charBBs = pseudo_data[label]['preds']
                        for char, (cx1, cy1, cx2, cy2) in charBBs.items():
                            charBB = np.array([cx1, cy1, cx2, cy1, cx2, cy2, cx1, cy2], dtype=np.int32).reshape((4, 2))
                            char_bboxes.append(charBB)
                            char_texts.append(char)
                            char_ignores.append(char in self.ignore_tags)

        data = {
            'wordBBs': np.array(bboxes),
            'texts': texts,
            'ignore_tags': ignores,
        }

        if pseudo:
            data['charBBs'] = [char_bboxes]
            data['texts'].extend(char_texts)
            data['ignore_tags'].extend(char_ignores)
        return data


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
    conf_name = 'icdar13_textfusenet'
    conf_path = conf_dir / f'{conf_name}.yaml'


    mode = 'train'
    main_cfg = OmegaConf.load(str(conf_path))
    base_cfg = OmegaConf.load(main_cfg[mode]['base'])
    main_cfg[mode]['dataset'] = base_cfg['dataset']

    ds_name = main_cfg[mode]['dataset']['name']
    data_path = Path(root) / main_cfg[mode]['root']
    root_path = data_path / main_cfg[mode]['dataset']['root_path']
    annot_path = data_path / main_cfg[mode]['dataset']['annot_path']

    ignore_tags = main_cfg[mode]['dataset']['args']['ignore_tags']
    filter_keys = main_cfg[mode]['dataset']['args']['filter_keys']
    tfms_cfg = main_cfg[mode]['transform']

    def collate_fn(batch):
        return tuple(zip(*batch))

    tfms_name = tfms_cfg.pop('name')
    tfms = eval(tfms_name)(**tfms_cfg)
    trn_ds = TextFuseNetICDAR13DS(root_path, annot_path, ignore_tags, filter_keys, transform=tfms)
    trn_dl = DataLoader(trn_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    label_names = trn_ds.label2cls


    cnt = 0
    show = 100
    vis = True
    if vis:
        debug = workspace / 'debug' / 'textfusenet' / 'icdar13'
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

                mask_save_p = osp.join(debug, f"draw_{cnt}.jpg")
                cv2.imwrite(mask_save_p, (draw_image*255.).astype(np.uint8))

        cnt += 1
        if cnt == show:
            break