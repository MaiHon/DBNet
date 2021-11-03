import os
import os.path as osp
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)


from pathlib import Path
import numpy as np
import json


import torch
from torch.utils.data import Dataset
from src.datasets.dbnet import DBNetBaseDS
from src.datasets.processing import DBNetCollateFN


class DBNetICDAR15DS(DBNetBaseDS):
    def __init__(self, root_p, annots_p, pre_processes, ignore_tags, filter_keys, transform=None, return_origin=False, **kwargs):
        self.root_p = root_p
        self.annots_p = annots_p
        self.ignore_tags = ignore_tags
        self.filter_keys = filter_keys

        super().__init__(annots_p, pre_processes, ignore_tags, filter_keys, transform, return_origin)


    def load_annot(self, annot):
        full_annot_p = Path(self.annots_p) / annot
        im_name = annot[3:-4]
        if osp.exists(osp.join(self.root_p, '{}.jpg'.format(im_name))):
            im_name = f'{im_name}.jpg'
        elif osp.exists(osp.join(self.root_p, '{}.png'.format(im_name))):
            im_name = f'{im_name}.png'
        elif osp.exists(osp.join(self.root_p, '{}.gif'.format(im_name))):
            im_name = f'{im_name}.gif'
        else:
            raise FileNotFoundError
        
        with open(str(full_annot_p), 'r') as f:
            contents = f.readlines()
        
        texts = []
        polys = []
        for content in contents:
            content = content.rstrip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            
            poly = content[:8]
            word = ''.join(content[8:])
            poly = np.array(list(map(int, poly))).reshape(-1, 2)
            
            polys.append(poly)
            texts.append(word)
            
        return im_name, texts, polys
        
    def prepare_data(self, annots_p):
        print("\nLoading GT...")
        print("It take some time...\n")

        data_list = []
        root_p = Path(self.root_p)
        for annot in sorted(os.listdir(self.annots_p)):
            im_name, texts, text_polys = self.load_annot(annot)
            img_fp = root_p / im_name

            text_polys = np.array(text_polys)
            transcripts = [t for t in texts if len(t) > 0]
            num_words = text_polys.shape[0]

            if num_words != len(transcripts):
                continue

            item = {}
            item['img_fp'] = str(img_fp)
            item['img_name'] = im_name
            item['text_polys'] = text_polys
            item['texts'] = transcripts
            item['ignore_tags'] = [x in self.ignore_tags for x in transcripts]

            data_list.append(item)
        return data_list
        
        
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
    conf_name = 'icdar15_dbnet'
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
    trn_ds = DBNetICDAR15DS(root_path, annot_path, pre_processes_args, ignore_tags, filter_keys)
    trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    visualize = True
    if visualize:
        debug = workspace / 'debug' / 'icdar15' / mode
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