import torch.nn as nn
from src.losses.base_loss import (
    BalanceCrossEntropyLoss,
    DiceLoss,
    MaskL1Loss
)


# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:56
# @Author  : zhoujun
class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)


    def forward(self, pred, batch):
        prob_maps = pred[:, 0, :, :]
        device = prob_maps.device
        
        prob_map_loss = self.bce_loss(prob_maps, batch['prob_map'].to(device), batch['prob_mask'].to(device))
        metrics = dict(prob_map_loss=prob_map_loss)
        loss_all = self.alpha * prob_map_loss
        
        if pred.size()[1] > 1:
            threshold_map_loss = self.l1_loss(pred[:, 1, :, :], batch['thresh_map'].to(device), batch['thresh_mask'].to(device))
            metrics['thresh_map_loss'] = threshold_map_loss

            approx_map_loss = self.dice_loss(pred[:, 2, :, :], batch['prob_map'].to(device), batch['prob_mask'].to(device))
            metrics['approx_map_loss'] = approx_map_loss            
            
            loss_all += self.beta * threshold_map_loss + approx_map_loss


        metrics['loss'] = loss_all
        return metrics
