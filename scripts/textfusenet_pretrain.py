import os
import os.path as osp
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)


import wandb
import hydra

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from src.models import Lightning_TextFuseNet


ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))

# @hydra.main(config_path='../configs/experiments', config_name='synthtext_textfusenet')
@hydra.main(config_path='../configs')
def main(cfgs):
    if 'experiments' in cfgs:
        cfgs = cfgs['experiments']

    seed_everything(cfgs['general']['seed'])
    if cfgs['experiment']['wandb']:
        wandb_logger = WandbLogger(project='[TextFuseNet]',
                                name=cfgs['experiment']['name'],
                                log_model=cfgs['experiment']['log_model'],
                                config=cfgs,
                                group=cfgs['experiment']['group'])
    else:
        wandb_logger = None


    trainer_cfg = {'gpus': cfgs['general']['gpus'],
                   'logger': wandb_logger,
                   'log_every_n_steps': 10,
                   'deterministic': True,
                   'precision': cfgs['general']['precision']}

    if 'valid' in cfgs:
        trainer_cfg['val_check_interval'] = cfgs['valid']['save_interval']
        trainer_cfg['num_sanity_val_steps'] = 2


    if 'max_steps' in cfgs['train']:
        trainer_cfg['max_steps'] = cfgs['train']['max_steps']
        trainer_cfg['val_check_interval'] = cfgs['valid']['save_interval']
    elif 'max_epochs' in cfgs['train']:
        trainer_cfg['max_epochs'] = cfgs['train']['max_epochs']
        trainer_cfg['limit_train_batches'] = cfgs['train']['limit_train_batches']


    if len(cfgs['general']['gpus']) > 1:
        trainer_cfg['num_nodes'] = 1
        trainer_cfg['accelerator'] = 'ddp'
        trainer_cfg['plugins'] = DDPPlugin(find_unused_parameters=True)

    checkpoint_dict = {
        'dirpath': f"{ROOT_DIR}/ckpts/textfusenet/{cfgs['experiment']['name']}",
        'save_last': True,
        'auto_insert_metric_name': True
    }


    checkpoint_dict['monitor'] = 'icdar_hmean'
    checkpoint_dict['filename'] = "[SynthText]-{icdar_hmean:.4f}-{det_hmean:.4f}"
    checkpoint_dict['save_top_k'] = cfgs['valid']['save_top_k']
    checkpoint_dict['every_n_train_steps'] = cfgs['valid']['save_interval']
    checkpoint_dict['mode'] = 'max'
    checkpoint_callback = ModelCheckpoint(**checkpoint_dict)


    pl_model = Lightning_TextFuseNet(cfgs, debug=False)
    pl_trainer = Trainer(**trainer_cfg, callbacks=[checkpoint_callback])
    if wandb_logger is not None:
        wandb_logger.watch(pl_model)

    fit_args = {
        'model': pl_model,
        'train_dataloaders': pl_model.trn_dl,
        'val_dataloaders': pl_model.val_dl
    }
    pl_trainer.fit(**fit_args)
    wandb.finish()


if __name__ == '__main__':
    main()