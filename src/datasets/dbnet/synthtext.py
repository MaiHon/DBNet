import os
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)

import numpy as np
import scipy.io as sio
from src.datasets.dbnet.base import DBNetBaseDS
from src.datasets.utils import polygon_validity_check


class DBNetSynthTextDS(DBNetBaseDS):
    def __init__(self, root_p, annots_p, pre_processes, ignore_tags, filter_keys, transform=None, return_origin=False, **kwargs):
        self.root_p = root_p
        self.annots_p = annots_p
        self.ignore_tags = ignore_tags
        self.filter_keys = filter_keys
        super().__init__(annots_p, pre_processes, ignore_tags, filter_keys, transform, return_origin)


    def prepare_data(self, annots_p):
        print("\nLoading GT...")
        print("It take some time...\n")
        self.annots = sio.loadmat(annots_p)
        self.iter = zip(self.annots['wordBB'][0], self.annots['imnames'][0], self.annots['txt'][0])

        data_list = []
        for wordBBs, in_name, texts in self.iter:
            image_fp = self.root_p / in_name[0]

            wordBBs = np.expand_dims(wordBBs, axis=2) if (wordBBs.ndim == 2) else wordBBs
            _, _, num_words = wordBBs.shape

            text_polys = wordBBs.reshape([8, num_words], order='F').T
            text_polys = text_polys.reshape(num_words, 4, 2)
            texts = [word for line in texts for word in line.split()]
            transcripts = [t for t in texts if len(t) > 0]

            if num_words != len(transcripts):
                continue

            text_polys, transcripts = polygon_validity_check(text_polys, transcripts)

            item = {}
            item['img_fp'] = str(image_fp)
            item['img_name'] = image_fp.stem
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
    conf_name = 'synthtext_pretrain'
    conf_path = conf_dir / f'{conf_name}.yaml'

    main_cfg = OmegaConf.load(str(conf_path))
    base_cfg = OmegaConf.load(main_cfg['train']['base'])
    main_cfg['train']['dataset'] = base_cfg['dataset']

    ds_name = main_cfg['train']['dataset']['name']
    data_path = Path(root) / main_cfg['train']['root']
    root_path = data_path / main_cfg['train']['dataset']['root_path']
    annot_path = data_path / main_cfg['train']['dataset']['annot_path']

    pre_processes_args = main_cfg['train']['dataset']['args']['pre_processes']
    ignore_tags = main_cfg['train']['dataset']['args']['ignore_tags']
    filter_keys = main_cfg['train']['dataset']['args']['filter_keys']

    collate_fn = None

    trn_ds = DBNetSynthTextDS(root_path, annot_path, pre_processes_args, ignore_tags, filter_keys)
    trn_dl = DataLoader(trn_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    visualize = False
    if visualize:
        debug = workspace / 'debug' / 'synthtext'
        if not osp.exists(debug):
            os.makedirs(debug)

    for idx, batch in tqdm(enumerate(trn_dl), total=len(trn_dl)):
        img = batch['img'][0]
        prob_map = batch['prob_map'][0]
        thresh_map = batch['thresh_map'][0]

        if visualize:
            img_fp = batch['img_fp'][0]
            img_name = img_fp.split(os.sep)[-1].split('.')[0]

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 12))
            axes[0].imshow(img)
            axes[0].axis('off')
            axes[0].set_title('Img')

            axes[1].imshow(prob_map)
            axes[1].axis('off')
            axes[1].set_title('Prob Map')

            axes[2].imshow(thresh_map)
            axes[2].axis('off')
            axes[2].set_title('Thresh Map')

            img_name = debug / (str(img_name) + '.png')
            plt.savefig(str(img_name))
            plt.close()