import os
import os.path as osp
import sys

add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)

import numpy as np
from pathlib import Path
import omegaconf
from omegaconf import ListConfig

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset, Subset

from torchvision import transforms
from src.datasets import *
from src.datasets.dbnet import *
from src.datasets.textfusenet import *
from src.datasets.textfusenet import CollateFN
from src.datasets.processing import DBNetCollateFN
from src.datasets.transform import BasicTransform


ROOT_DIR = Path(osp.abspath(osp.join(osp.abspath(__file__), osp.pardir, osp.pardir, osp.pardir)))


def get_dataset(data_root_dir, cfgs, model_type=None, img_transforms=None):
    if model_type is None:
        return None

    dataset_args = cfgs['dataset']['args']
    if 'transforms' in dataset_args and img_transforms is None:
        img_transforms = get_transforms(dataset_args['transforms'])

    ds_name = model_type + cfgs['dataset']['name']
    root_path = data_root_dir / str(cfgs['dataset']['root_path'])
    if isinstance(cfgs['dataset']['annot_path'], omegaconf.listconfig.ListConfig):
        annot_path = [data_root_dir /  Path(p) for p in cfgs['dataset']['annot_path']]
    else:
        annot_path = data_root_dir / Path(cfgs['dataset']['annot_path'])
    if annot_path == None or annot_path == '':
        return None

    ds = eval(ds_name)(root_p=root_path, annots_p=annot_path, transform=img_transforms, **dataset_args)
    return ds


def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']

        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list


def get_dataloader(data_root_dir, cfgs, model_type=None, distributed=False):
    if cfgs is None or model_type is None:
        return None

    if 'transform' in cfgs:
        tfms_cfgs = dict(**cfgs['transform'])
        tfms_namw = tfms_cfgs.pop('name')
        img_transforms = eval(tfms_namw)(**tfms_cfgs)
    else:
        img_transforms = None

    if not isinstance(cfgs['dataset'], list) and not isinstance(cfgs['dataset'], ListConfig):
        tot_ds = get_dataset(data_root_dir, cfgs, model_type, img_transforms)
    else:
        tot_ds = []
        for cfg, frac in zip(cfgs['dataset'], cfgs['frac']):
            ds = get_dataset(data_root_dir, cfg, model_type, img_transforms)
            tot_size = len(ds)
            indicies = np.random.choice(tot_size, int(tot_size*frac))
            ds = Subset(ds, indicies)

            tot_ds.append(ds)
        tot_ds = ConcatDataset(tot_ds)


    loader_cfgs = {**cfgs['loader']}
    if 'collate_fn' not in cfgs['loader'] or cfgs['loader']['collate_fn'] is None:
        loader_cfgs['collate_fn'] = None
    else:
        loader_cfgs['collate_fn'] = eval(cfgs['loader']['collate_fn'])()

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(tot_ds)
        cfgs['loader']['shuffle'] = False
        cfgs['loader']['pin_memory'] = True

    loader = DataLoader(dataset=tot_ds, sampler=sampler, **loader_cfgs)
    return loader