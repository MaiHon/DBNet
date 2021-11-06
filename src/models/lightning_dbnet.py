import os, sys
import os.path as osp

add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)

import wandb
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl

import torch
import torch.optim as optimizer
from torchinfo import summary

from src.modules import *
from src.losses import DBLoss
from src.datasets import get_dataloader
from src.evals import calc_icdar_det_metrics, calc_deteval_metrics, cal_text_score
from src.evals import QuadMetric, Scoring
from src.utils import visuailze_imgs, Seg2BoxOrPoly, denormalize


ROOT_DIR = Path(osp.abspath(osp.join(osp.abspath(__file__),
                                osp.pardir, # models
                                osp.pardir, # modules
                                osp.pardir)))


class Lightning_DBNet(pl.LightningModule):
    def __init__(self, cfgs, debug=False):
        super().__init__()
        self.cfgs = cfgs
        self.all_setup()
        self.automatic_optimization = True

        if debug:
            print("Config:\n")
            print(OmegaConf.to_yaml(cfgs))

            print('\nModel summary:\n')
            summary(self.model)

            print('\nOptimizer:\n', self.optim)
            print('\nCriterion:\n', self.criterion)
            print('\nTrain loader:\n', self.trn_dl)
            if self.val_dl is not None:
                print('\nValid loader:\n', self.val_dl)


    def configure_model(self):
        self.model = DBNet(self.cfgs['model'])

        if self.cfgs['model']['pretrained'] != '':
            lightning_dict = torch.load(str(ROOT_DIR / self.cfgs['model']['pretrained']), map_location='cpu')
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
        return self.optim


    def configure_criterion(self):
        self.criterion = DBLoss()


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
        self.configure_criterion()
        self.configure_optimizers()
        self.configure_dataloader()


        # post processing setup
        self.post_process = eval(self.cfgs['post_processing']['type'])(**self.cfgs['post_processing']['args'])
        self.is_output_polygon = self.cfgs['post_processing']['is_output_polygon']


        # scheduler setup
        self.cur_step = 0
        self.factor = self.cfgs['train']['scheduler']['factor']
        self.lr = self.cfgs['train']['optimizer']['args']['lr']


        # train epoch or step setup
        self.train_on_step = True
        if 'max_epochs' in self.cfgs['train']:
            self.max_step = self.cfgs['train']['max_epochs']
            self.cur_epoch = 0
            self.train_on_step = False
        elif 'max_steps' in self.cfgs['train']:
            self.max_step = self.cfgs['train']['max_steps']


        # evaluation setup
        self.scoring_metric = Scoring(2)
        self.val_metric_cls = QuadMetric(self.is_output_polygon)


        # wandb setup
        self.log_step = self.cfgs['experiment']['log_step']


        # visualize setup
        self.trn_show = self.cfgs['train']['show']
        self.trn_show_interval = self.cfgs['train']['show_interval']
        if 'valid' in self.cfgs:
            self.val_show = self.cfgs['valid']['show']
            self.val_show_interval = self.cfgs['valid']['show_interval']

        self.mean = self.cfgs['train']['dataset'][0]['dataset']['args']['transforms'][-1]['args']['mean']
        self.std = self.cfgs['train']['dataset'][0]['dataset']['args']['transforms'][-1]['args']['std']


    def forward(self, data):
        return self.model(data['img'])


    def update_lr(self, step):
        rate = np.power(1.0 - step / float(self.max_step), self.factor)
        for g in self.optim.param_groups:
            g['lr'] = rate * self.lr


    def on_train_epoch_start(self):
        self.trn_step = 0


    def on_train_epoch_end(self):
        if "max_epochs" in self.cfgs['train']:
            self.cur_epoch += 1
            self.update_lr(self.cur_epoch)


    def training_step(self, batch, batch_idx):
        if self.train_on_step:
            self.cur_step += 1
            self.update_lr(self.cur_step)

        self.trn_step += 1
        img = batch['img']
        preds = self.model(img)
        losses = self.criterion(preds, batch)


        # logging
        log_template = {
            "learning_rate": self.optim.param_groups[0]['lr'],
            "trn_prob_map_loss": losses['prob_map_loss'].item(),
            "trn_thresh_map_loss": losses['thresh_map_loss'].item(),
            "trn_approx_map_loss": losses['approx_map_loss'].item(),
            "trn_total_loss": losses['loss'].item(),
        }


        # calc metric
        score_prob_map = cal_text_score(preds[:, 0, :, :],
                                        batch['prob_map'], batch['prob_mask'],
                                        self.scoring_metric,
                                        thred=self.cfgs['post_processing']['args']['thresh'])
        acc = score_prob_map['Mean Acc']
        prob_map_iou = score_prob_map['Mean IoU']
        log_template["trn_acc"] = acc
        log_template["trn_prob_map_iou"] = prob_map_iou


        # visualize
        if self.trn_show and ((self.trn_step % self.trn_show_interval == 0)):
            boxes_batch, scores_batch = self.post_process(preds, batch, is_output_polygon=self.is_output_polygon, by_seg=True)

            if self.cfgs['experiment']['wandb']:
                denorm_img = denormalize(batch, self.mean, self.std)
                visualize_imgs = self.show(batch, preds)

                vis = {'trn_labels': wandb.Image(visualize_imgs[0]),
                       'trn_predictions': wandb.Image(visualize_imgs[1])}
                if 'text_polys' in batch:
                    polygons = self.post_process.draw_polygons(denorm_img*255., boxes_batch, batch['text_polys'], batch['ignore_tags'])
                    vis['trn_polygons'] = wandb.Image(polygons),
                self.logger.experiment.log(vis)


        self.log_dict(
            log_template,
            on_step=self.log_step, on_epoch=True, prog_bar=True, logger=True
        )

        return dict(loss=losses['loss'])


    def on_validation_epoch_start(self):
        self.avg_precision = 0
        self.avg_recall = 0
        self.avg_hmean = 0
        self.val_step = 0

        self.raw_metrics = []
        self.pred_bboxes_dict = {}
        self.gt_bboxes_dict = {}
        self.transcriptions_dict = {}


    def on_validation_epoch_end(self):
        metrics = self.val_metric_cls.gather_measure(self.raw_metrics)
        ic_result = calc_icdar_det_metrics(self.pred_bboxes_dict, self.gt_bboxes_dict, self.transcriptions_dict)
        de_result = calc_deteval_metrics(self.pred_bboxes_dict, self.gt_bboxes_dict, self.transcriptions_dict)

        log_template = {
            'icdar_hmean': ic_result['total']['hmean'],
            'icdar_recall': ic_result['total']['recall'],
            'icdar_precision': ic_result['total']['precision'],

            'det_hmean': de_result['total']['hmean'],
            'det_recall': de_result['total']['recall'],
            'det_precision': de_result['total']['precision'],

            'quad_hmean': metrics['fmeasure'].avg,
            'quad_recall': metrics['recall'].avg,
            'quad_precision': metrics['precision'].avg,
        }

        self.log_dict(
                log_template,
                on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return log_template


    def validation_step(self, batch, batch_idx):
        self.val_step += 1
        img = batch['img']
        preds = self.model(img)

        # post process
        boxes_batch, scores_batch = self.post_process(preds, batch, is_output_polygon=self.is_output_polygon)

        # calc score
        raw_metric = self.val_metric_cls.validate_measure(batch, (boxes_batch, scores_batch))
        self.raw_metrics.append(raw_metric)

        for idx, (pred_bboxes, pred_scores) in enumerate(zip(boxes_batch, scores_batch)):
            valid_pred_bboxes = []
            for i in range(pred_bboxes.shape[0]):
                if pred_scores[i] >= self.cfgs['post_processing']['args']['box_thresh']:
                    valid_pred_bboxes.append(pred_bboxes[i, :, :].astype(np.int))

            sample_idx = batch['img_name'][idx]
            gt_bboxes = batch['text_polys'][idx]
            gt_transcription = batch['texts'][idx]

            self.pred_bboxes_dict[sample_idx] = valid_pred_bboxes
            self.gt_bboxes_dict[sample_idx] = gt_bboxes
            self.transcriptions_dict[sample_idx] = gt_transcription


        if self.val_show and self.cfgs['experiment']['wandb'] and ((self.val_step % self.val_show_interval == 0) or self.val_step == 1):
            visualize_imgs = self.show(batch, preds)

            vis = {'val_prediction': wandb.Image(visualize_imgs[1])}
            if visualize_imgs[0] is not None:
                vis['val_labels'] = wandb.Image(visualize_imgs[0])

            polygons = self.post_process.draw_polygons(batch['ori_img'], boxes_batch, batch['text_polys'], batch['ignore_tags'])
            vis['val_polygons'] = wandb.Image(polygons)
            self.logger.experiment.log(vis)


    def show(self, batch, preds):
        show_labels, show_preds = visuailze_imgs(batch, preds)
        return show_labels, show_preds


if __name__ == "__main__":
    from pytorch_lightning import seed_everything

    seed_everything(822)
    cfg_path = ROOT_DIR / 'configs' / 'experiments' / 'synthtext_pretrain.yaml'

    main_cfg = OmegaConf.load(cfg_path)
    model = Lightning_DBNet(main_cfg, debug=True)