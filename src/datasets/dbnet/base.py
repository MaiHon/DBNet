import cv2
import copy
import numpy as np
import torch

from omegaconf import DictConfig
from torch.utils.data import Dataset
from src.datasets.processing import (
    IaaAugment,
    RandomCrop,
    ResizeShortSize,
    ProbabiltyMapGenerator,
    ThresholdMapGenerator
)


class DBNetBaseDS(Dataset):
    def __init__(self, data_path, pre_processes, ignore_tags, filter_keys, transform=None, return_origin=False):
        self.transform = transform
        self.ignore_tags = ignore_tags
        self.filter_keys = filter_keys
        self.return_origin = return_origin
        self.init_pre_process(pre_processes)
        self.data_list = self.prepare_data(data_path)

    
    def prepare_data(self, data_path):
        raise NotImplementedError
    
    
    def init_pre_process(self, pre_process_args):
        self.augs = []
        if pre_process_args is None: return
        else:
            for augment in pre_process_args:
                if 'args' not in augment:
                    args = {}
                else:
                    args = augment['args']
                
                
                if isinstance(args, (dict, DictConfig)):
                    cls = eval(augment['type'])(**args)
                else:
                    only_resize = augment.get('only_resize', None)
                    keep_ratio = augment.get('keep_ratio', None)
                    cls = eval(augment['type'])(args, only_resize, keep_ratio)
                self.augs.append(cls)
        
    
    def pre_process(self, data):
        for augment in self.augs:
            data = augment(data)
        return data
        
    
    def __len__(self):
        return len(self.data_list)
        
        
    def __getitem__(self, idx):
        data = copy.deepcopy(self.data_list[idx])
        im = cv2.imread(data['img_fp'], cv2.IMREAD_UNCHANGED)
        
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.return_origin:
            data['ori_img'] = torch.from_numpy(im.copy()).permute(2, 0, 1).float()
            
        data['img'] = im
        data['shape'] = im.shape[:2]
        data = self.pre_process(data)
        
        
        if self.transform:
            data['img'] = self.transform(data['img'])
            
            if 'prob_map' in data:
                data['prob_map'] = torch.from_numpy(data['prob_map'])
                data['thresh_map'] = torch.from_numpy(data['thresh_map'])
                data['prob_mask'] = torch.from_numpy(data['prob_mask'])
                data['thresh_mask'] = torch.from_numpy(data['thresh_mask'])
                
        data['text_polys'] = data['text_polys'].tolist()
        if len(self.filter_keys):
            data_dict = {}
            for k, v in data.items():
                if k not in self.filter_keys:
                    data_dict[k] = v
                    return data_dict
        else:
            return data