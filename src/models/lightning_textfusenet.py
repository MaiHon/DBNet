import os, sys
import os.path as osp
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)

import cv2
import wandb
import random
import numpy as np
from pathlib import Path
from omegaconf import ListConfig, OmegaConf
import pytorch_lightning as pl

import torch
import torch.optim as optimizer
import torch.optim.lr_scheduler as scheduler
from torchinfo import summary

from src.modules import *
from src.datasets import get_dataloader, get_dataset
from src.evals import calc_deteval_metrics, calc_icdar_det_metrics
from src.utils import final_prediction, denormalize_image, draw_boxes_masks, mask2bbox
from src.datasets.textfusenet import label2cls
# from modules.utils import ComposedTransformation, GeoTransformation, BasicTransformation


ROOT_DIR = Path(osp.abspath(osp.join(osp.abspath(__file__),
                                osp.pardir, # models
                                osp.pardir, # modules
                                osp.pardir)))


class Lightning_TextFuseNet(pl.LightningModule):
    def __init__(self, cfgs, debug=False):
        super().__init__()
        self.cfgs = cfgs
        self.debug = debug
        self.all_setup()
        self.automatic_optimization = True

        if debug:
            print("Config:\n")
            print(OmegaConf.to_yaml(cfgs))

            print('\nModel summary:\n')
            summary(self.model)

            print('\nOptimizer:\n', self.optim)
            print('\nTrain loader:\n', self.trn_dl)
            if self.val_dl is not None:
                print('\nValid loader:\n', self.val_dl)


    def configure_model(self):
        self.model = TextFuseNet(self.cfgs['model'])

        # TODO: pretrained load
        if self.cfgs['model']['pretrained'] != '':
            pretrain_p = ROOT_DIR / str(self.cfgs['model']['pretrained'])

            lightning_dict = torch.load(pretrain_p, map_location='cpu')
            model_dict = lightning_dict['state_dict']
            orig_dict = self.model.state_dict()

            new_model_dict = {k[6:]:v for k, v in model_dict.items() if k[6:] in orig_dict.keys()}
            orig_dict.update(new_model_dict)
            self.model.load_state_dict(orig_dict)
            print("SynthText Pretrained loaded.\n")


    def configure_optimizers(self):
        optim_name = self.cfgs['train']['optimizer']['name']
        optim_args = self.cfgs['train']['optimizer']['args']
        self.optim = getattr(optimizer, optim_name)(self.parameters(), **optim_args)


        sched_name = self.cfgs['train']['scheduler']['name']
        sched_args = self.cfgs['train']['scheduler']['args']
        self.sched = getattr(scheduler, sched_name)(self.optim, **sched_args)
        return {'optimizer': self.optim, 'lr_scheduler': self.sched}


    def configure_dataloader(self):
        distributed = self.cfgs['general']['distributed']

        data_root_dir = Path(osp.join(ROOT_DIR, osp.pardir, 'data'))
        self.trn_dl = get_dataloader(data_root_dir,
                                     self.cfgs['train'],
                                     self.cfgs['model']['name'],
                                     distributed)
        self.avg_trn_loss = 0
        self.trn_step = 0

        if 'valid' in self.cfgs:
            self.val_dl = get_dataloader(data_root_dir,
                                         self.cfgs['valid'],
                                         self.cfgs['model']['name'],
                                         distributed)
            self.avg_precision = 0
            self.avg_recall = 0
            self.avg_hmean = 0
            self.val_step = 0
        else:
            self.val_dl = None


    def all_setup(self):
        # configs load
        if isinstance(self.cfgs['train']['base'], str):
            train_data_cfg = OmegaConf.load(ROOT_DIR / self.cfgs['train']['base'])
            train_data_cfg = [train_data_cfg]
        else:
            train_data_cfg = []
            for base in self.cfgs['train']['base']:
                base = ROOT_DIR / base
                train_data_cfg.append(OmegaConf.load(base))
        self.cfgs['train']['dataset'] = train_data_cfg

        if 'valid' in self.cfgs:
            valid_data_cfg = OmegaConf.load(ROOT_DIR / self.cfgs['valid']['base'])
            self.cfgs['valid']['dataset'] = valid_data_cfg['dataset']

        model_cfg_p = ROOT_DIR / self.cfgs['model']
        model_cfg = OmegaConf.load(model_cfg_p)
        self.cfgs['model'] = model_cfg


        # model, criterion, optimizer, dataloader setup
        self.configure_model()
        self.configure_optimizers()
        self.configure_dataloader()


        # define max_iter & scheduler factor
        if 'max_epochs' in self.cfgs['train']:
            self.max_step = self.cfgs['train']['max_epochs']
            self.cur_epoch = 0
        elif 'max_steps' in self.cfgs['train']:
            self.max_step = self.cfgs['train']['max_steps']

        self.lr = self.cfgs['train']['optimizer']['args']['lr']
        self.cur_step = 0
        self.log_step = self.cfgs['experiment']['log_step']


        self.mean = self.cfgs['train']['transform']['mean']
        self.std = self.cfgs['train']['transform']['std']


        if 'valid' in self.cfgs:
            self.val_show = self.cfgs['valid']['show']
            self.val_show_interval = self.cfgs['valid']['show_interval']


    def forward(self, data): # for inference only
        return self.model(data['img'])


    def validation_step(self, batch, batch_idx):
        self.val_step += 1

        images, targets = batch
        images = [image.to(self.device) for image in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            _, preds = self.model(images)

        pred_bboxes_dict = {}
        gt_bboxes_dict = {}
        transcriptions_dict = {}
        if self.logger:
            logging_images = []

            for i in range(len(preds)):
                r_masks, r_boxes, r_labels = final_prediction(preds[i], total_labels=label2cls)
                valid_boxes = [box for box, label in zip(r_boxes, r_labels) if label == 'word']
                valid_masks = [mask for mask, label in zip(r_masks, r_labels) if label == 'word']
                valid_labels = [label for label in r_labels if label == 'word']

                # icdar13 metric & deteval metric
                boxes_from_masks = []
                for mask in valid_masks:
                    box, ret = mask2bbox(mask)
                    if ret:
                        boxes_from_masks.append(box)

                original_image = images[i].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                original_image = denormalize_image(original_image, self.mean, self.std)
                # draw_box_image = draw_boxes_masks(original_image, r_masks, r_boxes, r_labels)
                draw_box_image = draw_boxes_masks(original_image, np.array(valid_masks), boxes_from_masks, valid_labels, polygon=False)
                draw_box_image = cv2.cvtColor(draw_box_image, cv2.COLOR_BGR2RGB)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                sample_idx = targets[i]['img_indx'].detach().cpu().item()
                gt_bboxes = targets[i]['boxes'].detach().cpu().numpy()
                gt_transcription = ['text' for _ in range(len(gt_bboxes))]

                pred_bboxes_dict[sample_idx] = boxes_from_masks
                gt_bboxes_dict[sample_idx] = gt_bboxes
                transcriptions_dict[sample_idx] = gt_transcription

                logging_images.append(wandb.Image(draw_box_image))
                if self.debug:
                    save_path = os.path.join(ROOT_DIR, 'debug/textfusenet/valid_check')
                    cv2.imwrite(os.path.join(save_path, 'draw_box_image.jpg'), cv2.cvtColor(draw_box_image, cv2.COLOR_BGR2RGB))

            self.logger.experiment.log({'predictions': logging_images})

            ic_result = calc_icdar_det_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict)
            de_result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict)

            log_template = {
                'icdar_hmean': ic_result['total']['hmean'],
                'icdar_precision': ic_result['total']['precision'],
                'icdar_recall': ic_result['total']['recall'],

                'deteval_hmean': de_result['total']['hmean'],
                'deteval_precision': de_result['total']['precision'],
                'deteval_recall': de_result['total']['recall'],
            }
            self.log_dict(
                log_template,
                on_step=self.log_step, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
            )
        return log_template


    def training_step(self, batch, batch_idx): # training step & manual optimization & lr scheduling
        if "max_steps" in self.cfgs['train']:
            self.cur_step += 1
            self.sched.step()

        # forward path
        images, targets = batch
        images = [image.to(self.device) for image in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        losses, _ = self.model(images, targets)
        loss = sum(loss for loss in losses.values())

        if (self.debug and self.logger) and (self.trn_step % 10 == 0):
            COLORS = np.random.uniform(0, 255, size=(64, 3))

            logging_images = []
            for i in range(len(images)):
                image_np = images[i].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

                w, h = targets[i]['shape']
                image_np = cv2.resize(image_np, (h, w))
                image_np = denormalize_image(image_np, self.mean, self.std)

                labels = targets[i]['labels']
                for i, box in enumerate(targets[i]['boxes']):
                    color = COLORS[random.randrange(0, len(COLORS))]
                    x1, y1, x2, y2 = [int(coord.item()) for coord in box]
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), color=color, thickness=2, lineType=cv2.LINE_AA)

                    label = labels[i].item()
                    if label in label2cls:
                        cv2.putText(image_np, label2cls[labels[i].item()], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,
                                thickness=1, lineType=cv2.LINE_AA)
                    else:
                        cv2.putText(image_np, 'special', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,
                                thickness=1, lineType=cv2.LINE_AA)

                logging_images.append(wandb.Image(image_np))
                if self.debug:
                    save_path = os.path.join('/workspace/ocr-detector2/debug/training_test')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    cv2.imwrite(os.path.join(save_path, f'{str(self.cur_step).zfill(6)}_draw_box_image.jpg'), cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

            if self.logger:
                self.logger.experiment.log({'training_samples': logging_images})


        # optimization
        self.cur_step += 1
        self.trn_step += 1


        self.avg_trn_loss += loss.item()

        # logging
        log_template = {
            "learning_rate": self.optim.param_groups[0]['lr'],
            "trn_loss_classifier": losses['loss_classifier'].item(),
            "trn_loss_box_reg": losses['loss_box_reg'].item(),
            "trn_loss_mask": losses['loss_mask'].item(),
            "trn_loss_seg": losses['loss_seg'].item(),
            "trn_loss_objectness": losses['loss_objectness'].item(),
            "trn_loss_rpn_box_reg": losses['loss_rpn_box_reg'].item(),
            "trn_total_loss": loss.item(),
        }


        self.log_dict(
            log_template,
            on_step=self.log_step, on_epoch=True, prog_bar=True, logger=True
        )

        return loss


    def on_train_epoch_start(self):
        self.avg_trn_loss = 0
        self.trn_step = 0

        min_size = self.cfgs['train']['scaling']['min_size']
        max_size = self.cfgs['train']['scaling']['max_size']
        min_size = list(min_size) if isinstance(min_size, ListConfig) else min_size
        max_size = list(max_size) if isinstance(min_size, ListConfig) else max_size
        self.model.transform = GeneralizedRCNNTransform(min_size, max_size)
        self.model.train()


    def on_valid_epoch_start(self):
        min_size = self.cfgs['valid']['scaling']['min_size']
        max_size = self.cfgs['valid']['scaling']['max_size']
        min_size = list(min_size) if isinstance(min_size, ListConfig) else min_size
        max_size = list(max_size) if isinstance(min_size, ListConfig) else max_size
        self.model.transform = GeneralizedRCNNTransform(min_size, max_size)
        self.model.eval()

    def on_valid_epoch_end(self):
        torch.cuda.empty_cache()


    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

        if "max_epochs" in self.cfgs['train']:
            self.cur_epoch += 1
            self.sched.step()



if __name__ == '__main__':
    import wandb
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint


    seed_everything(42)
    cfg_path = ROOT_DIR / 'configs' / 'experiments' / 'synthtext_textfusenet.yaml'
    cfgs = OmegaConf.load(cfg_path)
    trainer_cfg = {'gpus': cfgs['general']['gpus'],
                   'precision': cfgs['general']['precision']}

    trainer_cfg['max_epochs'] = 1
    if 'valid' in cfgs:
        trainer_cfg['check_val_every_n_epoch'] = 10
        trainer_cfg['num_sanity_val_steps'] = 2

    checkpoint_dict = {
        'dirpath': f"{ROOT_DIR}/ckpts/textfusenet/{cfgs['experiment']['name']}",

        'save_last': True,
        'auto_insert_metric_name': True
    }

    if 'valid' in cfgs:
        checkpoint_dict['monitor'] = 'icdar_hmean'
        checkpoint_dict['filename'] = "{step:06d}-{icdar_hmean:.4f}-{deteval_hmean:.4f}"
        checkpoint_dict['save_top_k'] = cfgs['valid']['save_top_k']
        checkpoint_dict['every_n_epochs'] = cfgs['valid']['save_interval']
        checkpoint_dict['mode'] = "max"
    else:
        checkpoint_dict['monitor'] = 'trn_total_loss'
        checkpoint_dict['save_top_k'] = cfgs['train']['save_top_k']
        checkpoint_dict['every_n_train_steps'] = cfgs['train']['save_interval']
        checkpoint_dict['mode'] = "min"
    checkpoint_callback = ModelCheckpoint(**checkpoint_dict)

    pl_model = Lightning_TextFuseNet(cfgs, debug=True)
    pl_trainer = Trainer(**trainer_cfg,
                         log_every_n_steps=10,
                         callbacks=[checkpoint_callback],
                         deterministic=True)

    fit_args = {
        'model': pl_model,
        'train_dataloaders': pl_model.trn_dl,
        'val_dataloaders': pl_model.val_dl
    }
    pl_trainer.fit(**fit_args)
    wandb.finish()