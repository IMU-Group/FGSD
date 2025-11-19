import torch
from torch import nn
import torch.nn.functional as F


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

    
class DiceLoss(nn.Module):
    def __init__(self, apply_sigmoid=True, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, pred, target):
        if self.apply_sigmoid:
            pred = F.sigmoid(pred)
            
        numerator = 2 * torch.sum(pred * target) + self.smooth
        denominator = torch.sum(pred + target) + self.smooth
        return 1 - numerator / denominator
    
def alpha_loss(pred_logits, gt_alpha):
    # smooth_l1_loss
    loss = F.smooth_l1_loss(pred_logits, gt_alpha, reduction='none')
    shadow_area = (gt_alpha > 0.02).float()
    loss = (loss * shadow_area).sum() / (shadow_area.sum() + 1e-6)
    return loss 

def area_loss(pred_logits, gt_alpha):
    # 防止在不存在shadow的地方预测出shadow
    loss = F.l1_loss(pred_logits, gt_alpha, reduction='none')
    none_area = (gt_alpha <0.01).float()
    shadow_area = (gt_alpha > 0).float()
    loss = (loss * none_area).sum() / (none_area.sum() + 1e-6)
    loss = loss / ((shadow_area.sum() + 1e-6) / (none_area.sum() + 1e-6))
    return loss

def smooth_loss(pred, gt):
    # 计算 alpha 通道相邻像素之间的梯度
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_gt = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    dy_gt = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    
    # 计算平滑损失
    smooth_loss = torch.mean(torch.relu(dx_pred - dx_gt)) + torch.mean(torch.relu(dy_pred - dy_gt))
    
    return smooth_loss

def RMSELoss(pred, gt):
    return torch.sqrt(F.mse_loss(pred, gt))