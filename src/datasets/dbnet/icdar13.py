import os
import os.path as osp
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)

import cv2
from pathlib import Path
import numpy as np


from src.datasets.dbnet import DBNetBaseDS
from src.datasets.processing import DBNetCollateFN


class DBNetICDAR13DS(DBNetBaseDS):
    def __init__(self, root_p, annots_p, pre_processes, ignore_tags, filter_keys, transform=None, return_origin=False, split='train', **kwargs):
        self.root_p = root_p
        self.annots_p = annots_p
        self.ignore_tags = ignore_tags
        self.filter_keys = filter_keys
        self.split = split


        super().__init__(annots_p, pre_processes, ignore_tags, filter_keys, transform, return_origin)


    def prepare_data(self, annots_p):
        print("\nLoading GT...")
        print("It take some time...\n")

        annots_path = Path(annots_p)
        annot_names = sorted(os.listdir(str(annots_path)))

        data_list = []
        root_p = Path(self.root_p)
        for annot_name in annot_names:
            annot_path = annots_path / annot_name
            annot = self.get_annotation(str(annot_path))


            if self.split == "train":
                img_name = annot_name.split('_')[-1].split('.')[0] + ".jpg"
            else:
                img_name = "_".join(annot_name.split('_')[1:])
                img_name = img_name.split('.')[0] + ".jpg"


            image_fp = root_p / img_name
            if len(annot['text_polys']) > 0:
                item = {'img_fp': str(image_fp), 'img_name': image_fp.stem}
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


if __name__ == '__main__':
    import os
    import os.path as osp
    from tqdm import tqdm
    from pathlib import Path
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from src.datasets.processing import DBNetCollateFN
    
    import matplotlib.pyplot as plt

    root = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir, osp.pardir))
    workspace = Path(root) / 'workspace'
    conf_dir  = workspace / 'configs' / 'experiments'
    conf_name = 'icdar13_dbnet'
    conf_path = conf_dir / f'{conf_name}.yaml'

    mode = 'valid'
    main_cfg = OmegaConf.load(str(conf_path))
    base_cfg = OmegaConf.load(main_cfg[mode]['base'])
    main_cfg[mode]['dataset'] = base_cfg['dataset']

    ds_name = main_cfg[mode]['dataset']['name']
    data_path = Path(root) / main_cfg[mode]['root']
    root_path = data_path / main_cfg[mode]['dataset']['root_path']
    annot_path = data_path / main_cfg[mode]['dataset']['annot_path']

    pre_processes_args = main_cfg[mode]['dataset']['args']['pre_processes']
    ignore_tags = main_cfg[mode]['dataset']['args']['ignore_tags']
    filter_keys = main_cfg[mode]['dataset']['args']['filter_keys']
    
    collate_fn = None
    trn_ds = DBNetICDAR13DS(root_path, annot_path, pre_processes_args, ignore_tags, filter_keys, split=mode)
    trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    visualize = True
    if visualize:
        debug = workspace / 'debug' / 'icdar13' / mode
        if not osp.exists(debug):
            os.makedirs(debug)
    
    for idx, batch in tqdm(enumerate(trn_dl), total=len(trn_dl)):
        vis_list = []
        
        img = batch['img'][0]
        if mode not in ['valid', 'test']:
            prob_map = batch['prob_map'][0]
            thresh_map = batch['thresh_map'][0]
            
            vis_list.append(("Thresh Map", thresh_map))
            vis_list.append(("Prob Map", prob_map))
        vis_list.append(("Img", img))
        
        if visualize:
            img_fp = batch['img_fp'][0]
            img_name = img_fp.split(os.sep)[-1].split('.')[0]
            
            if len(vis_list) > 1:
                fig, axes = plt.subplots(nrows=1, ncols=len(vis_list), figsize=(15, 12))
                for idx, vis in enumerate(vis_list):
                    axes[idx].imshow(vis[1])
                    axes[idx].axis('off')
                    axes[idx].set_title(vis[0])
            else:
                plt.imshow(vis_list[0][1])
                plt.axis('off')
                plt.title(vis_list[0][0])
                
            img_name = debug / (str(img_name) + '.png')
            plt.savefig(str(img_name))
            plt.close()