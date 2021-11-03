import os
import sys

add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F

import modules


class DBNet(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(DBNet, self).__init__()

        self.backbone = getattr(modules, cfg['backbone'])(**cfg.get('backbone_args', {}))
        self.neck = modules.FPN(**cfg.get('fpn_args', {}))
        self.head = modules.DBHead(**cfg.get('dbhead_args', {}), mode=mode)

    def forward(self, x):
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        return y


if __name__ == '__main__':
    import yaml
    import os.path as osp
    from pathlib import Path
    from torchinfo import summary
    from omegaconf import OmegaConf

    root = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    root = Path(root)
    conf_dir  = root / 'configs' / 'experiments'
    conf_name = 'synthtext_pretrain'

    conf_path = conf_dir / f'{conf_name}.yaml'
    print(conf_path)
    print(osp.exists(conf_path))
    
    with open(conf_path, 'r') as f:
        cfgs = yaml.safe_load(f)
    model_cfg_p = cfgs['model']
    model_cfg = OmegaConf.load(model_cfg_p)
    
    print('model config:\n', model_cfg)
    model = DBNet(model_cfg)

    summary(model)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    fake_inp = torch.rand(2, 3, 688, 688).float().to(device)
    fake_out = model(fake_inp)

    print('=' * 15, 'DBNet Ouput Information', '=' * 15)
    for k, v in fake_out.items():
        print(f"{k} : {v.shape}")
        print('-' * 58)
    print('=' * 58)