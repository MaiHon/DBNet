import os
import os.path as osp
import sys
add_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(add_dir)
add_dir = f'{os.path.sep}'.join(add_dir.split(os.path.sep)[:-1])
sys.path.append(add_dir)

import cv2
import json
import torch
import wandb
import hydra
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from omegaconf import OmegaConf

from src.modules import DBNet
from src.datasets import get_dataloader
from src.utils.visualize import Seg2BoxOrPoly, visuailze_imgs
from src.evals import (
    QuadMetric,
    calc_deteval_metrics,
    calc_icdar_det_metrics,
)

import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = Path(osp.abspath(osp.join(osp.dirname(__file__), osp.pardir)))


def evaluation(cfgs, model, device):
    # prepare data loader & scoring metrics & visualization
    data_root_dir = Path(osp.join(ROOT_DIR, osp.pardir, 'data'))
    test_data_cfg = OmegaConf.load(ROOT_DIR / cfgs['test']['base'])
    cfgs['test']['dataset'] = test_data_cfg['dataset']

    test_dl = get_dataloader(data_root_dir, cfgs['test'], cfgs['model']['name'], distributed=False)

    post_processing = eval(cfgs['post_processing']['type'])(**cfgs['post_processing']['args'])
    metric_cls = QuadMetric(is_output_polygon=cfgs['post_processing']['is_output_polygon'])

    total_results = {}
    raw_metrics = []
    pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict = dict(), dict(), dict()
    with torch.no_grad():
        with tqdm(test_dl, total=len(test_dl), unit='batch') as test_bar:
            test_bar.set_description(f'Evaluation of {cfgs["model"]["backbone"]}')

            for batch_num, batch in enumerate(test_bar):
                img = batch['img'].to(device)
                preds = model(img)
                boxes_batch, scores_batch = post_processing(preds, batch, is_output_polygon=cfgs['post_processing']['is_output_polygon'])

                json_file = {'paragraphs': {}, 'words': {}}
                if cfgs['experiment']['score']:
                    raw_metric = metric_cls.validate_measure(batch, (boxes_batch, scores_batch), box_thresh=cfgs['post_processing']['args']['box_thresh'])
                    raw_metrics.append(raw_metric)

                    for idx, (pred_bboxes, pred_scores) in enumerate(zip(boxes_batch, scores_batch)):
                        valid_pred_bboxes = []
                        for i in range(pred_bboxes.shape[0]):
                            if pred_scores[i] >= cfgs['post_processing']['args']['box_thresh']:
                                box = pred_bboxes[i, :, :].astype(np.int)
                                if cv2.contourArea(box) < 5: continue

                                valid_pred_bboxes.append(box)

                        sample_idx = batch['img_name'][idx]
                        gt_bboxes = batch['text_polys'][idx]
                        gt_transcription = batch['texts'][idx]

                        pred_bboxes_dict[sample_idx] = valid_pred_bboxes
                        gt_bboxes_dict[sample_idx] = gt_bboxes
                        transcriptions_dict[sample_idx] = gt_transcription

                    polygons = post_processing.draw_polygons(batch['ori_img'], boxes_batch, batch['text_polys'], batch['ignore_tags'])
                    if cfgs['experiment']['save_demo_image'] or cfgs['experiment']['wandb']:

                        if cfgs['experiment']['save_demo_image']:
                            save_dir = str(ROOT_DIR / 'results' / cfgs['experiment']['name'])
                            if not osp.exists(save_dir):
                                os.makedirs(save_dir)

                            save_name = Path(batch['img_fp'][0]).stem
                            save_path = Path(save_dir) / str(save_name + '.jpg')
                            cv2.imwrite(str(save_path), cv2.cvtColor(polygons.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB))

                            save_path = Path(save_dir) / str(save_name + '_pred.jpg')
                            cv2.imwrite(str(save_path), preds[0].permute(1, 2, 0).detach().cpu().numpy() * 255)


                        if cfgs['experiment']['wandb']:
                            show_label, show_preds = visuailze_imgs(batch, preds)
                            wandb.log({
                                'val_polygons': wandb.Image(polygons),
                                'val_preds': wandb.Image(show_preds)
                            })

    if cfgs['experiment']['score']:
        qu_result = metric_cls.gather_measure(raw_metrics)
        ic_result = calc_icdar_det_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict)
        de_result = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict)

        save_keys = ['precision', 'recall', 'hmean', 'gt_bboxes', 'det_bboxes']
        for sample in ic_result['per_sample']:
            total_results[sample] = {}
            for key in save_keys:
                total_results[sample][key] = ic_result['per_sample'][sample][key]


        print("\n")
        print("="*16, "Quad Metric", "="*16)
        print(f"Average Hmean/Fmeasure: {qu_result['fmeasure'].avg}")
        print(f"Average Recall: {qu_result['recall'].avg}")
        print(f"Average Precision: {qu_result['precision'].avg}")

        print("="*16, "ICDAR Metric", "="*16)
        print(f"Hmean/Fmeasure: {ic_result['total']['hmean']}")
        print(f"Recall: {ic_result['total']['recall']}")
        print(f"Precision: {ic_result['total']['precision']}")


        print("="*16, "DetEval Metric", "="*16)
        print(f"Hmean/Fmeasure: {de_result['total']['hmean']}")
        print(f"Recall: {de_result['total']['recall']}")
        print(f"Precision: {de_result['total']['precision']}")
    print("Done.")



@hydra.main(config_path='../configs/evals', config_name='icdar15_dbnet')
def main(cfgs):
    if 'evals' in cfgs:
        cfgs = cfgs['evals']

    if 'dbnet' in cfgs:
        cfgs = cfgs['dbnet']

    if cfgs['experiment']['wandb']:
        wandb.init(project='[DBNet]',
                    name=cfgs['experiment']['name'],
                    config=cfgs,
                    group=cfgs['experiment']['group'])


    # model prepare
    device = cfgs['test']['device']
    model_cfg = OmegaConf.load(ROOT_DIR / cfgs['model'])
    cfgs['model'] = model_cfg
    model = DBNet(model_cfg, mode='eval')

    full_ckpt_dir = str(ROOT_DIR / cfgs['finetuned'])
    if cfgs['finetuned'] != '' and os.path.isfile(full_ckpt_dir):
        state_dict = torch.load(full_ckpt_dir, map_location='cpu')
        model_dict = state_dict['state_dict']

        new_model_dict = OrderedDict()
        for k, v in model_dict.items():
            new_k = k[6:]
            new_model_dict[new_k] = v

        model.load_state_dict(new_model_dict)
        print(f"{cfgs['finetuned']} Weight loaded.\n")
        model = model.to(device)
        model.eval()
        evaluation(cfgs, model, device)

    elif cfgs['finetuned'] != '' and os.path.isdir(full_ckpt_dir):
        files = sorted(os.listdir(full_ckpt_dir))
        for file in files:
            full_file_path = os.path.join(str(ROOT_DIR / cfgs['finetuned']), file)

            state_dict = torch.load(full_file_path, map_location='cpu')
            model_dict = state_dict['state_dict']

            new_model_dict = OrderedDict()
            for k, v in model_dict.items():
                new_k = k[6:]
                new_model_dict[new_k] = v

            model.load_state_dict(new_model_dict)
            print(f"{full_file_path} Weight loaded.\n")
            model = model.to(device)
            model.eval()

            evaluation(cfgs, model, device)
            print("\n")


if __name__ == '__main__':
    main()