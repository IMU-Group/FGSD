import torch
import numpy as np
from collections import OrderedDict
import pandas as pd
import os
from tqdm import tqdm
import cv2
from utils.misc import split_np_imgrid, get_np_imgrid
import pydensecrf.densecrf as dcrf


def cal_ber(tn, tp, fn, fp):
    return  0.5*(fp/(tn+fp) + fn/(fn+tp))

def cal_acc(tn, tp, fn, fp):
    return (tp + tn) / (tp + tn + fp + fn)


def get_binary_classification_metrics(pred, gt, threshold=None,soft_pred=None,soft_gt=None,soft_th=0.05):
    if threshold is not None:
        gt = (gt > threshold)
        pred = (pred > threshold)
    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(np.logical_not(gt), np.logical_not(pred)).sum()
    FN = np.logical_and(gt, np.logical_not(pred)).sum()
    FP = np.logical_and(np.logical_not(gt), pred).sum()
    BER = cal_ber(TN, TP, FN, FP)
    ACC = cal_acc(TN, TP, FN, FP)
    
    # ----------------soft----------------
    soft_gt = soft_gt / 255
    soft_pred = soft_pred / 255
    soft_rmse = np.sqrt(np.mean((soft_gt - soft_pred)**2))
    soft_wse = compute_wse_torch(torch.tensor(soft_pred), torch.tensor(soft_gt))
    soft_gt =np.where(soft_gt>soft_th,1,0)
    soft_pred = np.where(soft_pred>soft_th,1,0)
    # *****************soft-soft************
    soft_soft_TP = np.logical_and(soft_gt, soft_pred).sum()
    soft_soft_TN = np.logical_and(np.logical_not(soft_gt), np.logical_not(soft_pred)).sum()
    soft_soft_FN = np.logical_and(soft_gt, np.logical_not(soft_pred)).sum()
    soft_soft_FP = np.logical_and(np.logical_not(soft_gt), soft_pred).sum()
    soft_soft_BER = cal_ber(soft_soft_TN, soft_soft_TP, soft_soft_FN, soft_soft_FP)
    soft_soft_ACC = cal_acc(soft_soft_TN, soft_soft_TP, soft_soft_FN, soft_soft_FP)
    # *****************soft-01************
    soft_01_TP = np.logical_and(soft_pred, gt).sum()
    soft_01_TN = np.logical_and(np.logical_not(soft_pred), np.logical_not(gt)).sum()
    soft_01_FN = np.logical_and(soft_pred, np.logical_not(gt)).sum()
    soft_01_FP = np.logical_and(np.logical_not(soft_pred), gt).sum()
    soft_01_BER = cal_ber(soft_01_TN, soft_01_TP, soft_01_FN, soft_01_FP)
    soft_01_ACC = cal_acc(soft_01_TN, soft_01_TP, soft_01_FN, soft_01_FP)
    # *****************01-soft************
    
    return OrderedDict( [('TP', TP),
                        ('TN', TN),
                        ('FP', FP),
                        ('FN', FN),
                        ('BER', BER),
                        ('ACC', ACC),
                        ('soft_rmse', soft_rmse),
                        ('soft_wse', soft_wse),
                        ('soft_soft_TP', soft_soft_TP),
                        ('soft_soft_TN', soft_soft_TN),
                        ('soft_soft_FP', soft_soft_FP),
                        ('soft_soft_FN', soft_soft_FN),
                        ('soft_soft_BER', soft_soft_BER),
                        ('soft_soft_ACC', soft_soft_ACC),
                        ('soft_01_TP', soft_01_TP),
                        ('soft_01_TN', soft_01_TN),
                        ('soft_01_FP', soft_01_FP),
                        ('soft_01_FN', soft_01_FN),
                        ('soft_01_BER', soft_01_BER),
                        ('soft_01_ACC', soft_01_ACC),
                        ])


def evaluate(res_root, pred_id, gt_id, nimg, nrow, pred_soft_id=None, gt_soft_id=None):
    img_names  = os.listdir(res_root)
    score_dict = OrderedDict()

    for img_name in tqdm(img_names, disable=False):
        im_grid_path = os.path.join(res_root, img_name)
        im_grid = cv2.imread(im_grid_path)
        ims = split_np_imgrid(im_grid, nimg, nrow)
        pred = ims[pred_id]
        gt = ims[gt_id]
        pred_soft = ims[pred_soft_id]
        pred_gt = ims[gt_soft_id]
        score_dict[img_name] = get_binary_classification_metrics(pred,
                                                                 gt,
                                                                 125,soft_pred=pred_soft,soft_gt=pred_gt)
            
    df = pd.DataFrame(score_dict)
    df['ave'] = df.mean(axis=1)

    tn = df['ave']['TN']
    tp = df['ave']['TP']
    fn = df['ave']['FN']
    fp = df['ave']['FP']

    pos_err = (1 - tp / (tp + fn)) * 100
    neg_err = (1 - tn / (tn + fp)) * 100
    ber = (pos_err + neg_err) / 2
    acc = (tn + tp) / (tn + tp + fn + fp)
    
    soft_soft_tn = df['ave']['soft_soft_TN']
    soft_soft_tp = df['ave']['soft_soft_TP']
    soft_soft_fn = df['ave']['soft_soft_FN']
    soft_soft_fp = df['ave']['soft_soft_FP']
    soft_soft_pos_err = (1 - soft_soft_tp / (soft_soft_tp + soft_soft_fn)) * 100
    soft_soft_neg_err = (1 - soft_soft_tn / (soft_soft_tn + soft_soft_fp)) * 100
    soft_soft_ber = (soft_soft_pos_err + soft_soft_neg_err) / 2
    soft_soft_acc = (soft_soft_tn + soft_soft_tp) / (soft_soft_tn + soft_soft_tp + soft_soft_fn + soft_soft_fp)
    
    soft_01_tn = df['ave']['soft_01_TN']
    soft_01_tp = df['ave']['soft_01_TP']
    soft_01_fn = df['ave']['soft_01_FN']
    soft_01_fp = df['ave']['soft_01_FP']
    soft_01_pos_err = (1 - soft_01_tp / (soft_01_tp + soft_01_fn)) * 100
    soft_01_neg_err = (1 - soft_01_tn / (soft_01_tn + soft_01_fp)) * 100
    soft_01_ber = (soft_01_pos_err + soft_01_neg_err) / 2
    soft_01_acc = (soft_01_tn + soft_01_tp) / (soft_01_tn + soft_01_tp + soft_01_fn + soft_01_fp)

    soft_rmse = df['ave']['soft_rmse']
    soft_wse = df['ave']['soft_wse']
    return {
        'pos_err': pos_err,
        'neg_err': neg_err,
        'ber': ber,
        'acc': acc,
        'soft_soft_pos_err': soft_soft_pos_err,
        'soft_soft_neg_err': soft_soft_neg_err,
        'soft_soft_ber': soft_soft_ber,
        'soft_soft_acc': soft_soft_acc,
        'soft_01_pos_err': soft_01_pos_err,
        'soft_01_neg_err': soft_01_neg_err,
        'soft_01_ber': soft_01_ber,
        'soft_01_acc': soft_01_acc,
        'soft_rmse': soft_rmse,
        'soft_wse': soft_wse
    }



def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')




###############################################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, weight=1):
        self.sum += val * weight
        self.count += weight

    def average(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count

    def clear(self):
        self.sum = 0
        self.count = 0

def compute_cm_torch(y_pred, y_label, n_class):
    mask = (y_label >= 0) & (y_label < n_class)
    hist = torch.bincount(n_class * y_label[mask] + y_pred[mask],
                          minlength=n_class**2).reshape(n_class, n_class)
    return hist

def compute_RMSE_torch(y_pred, y_label):
    return torch.sqrt(torch.mean((y_pred - y_label)**2))

def compute_wse_torch(pred_alpha, gt_alpha,soft_th=0.05):
    # Define shadow and non-shadow regions for both pred_alpha and gt_alpha
    shadow_mask = gt_alpha > soft_th
    non_shadow_mask = gt_alpha <= soft_th

    # Compute RMSE in shadow and non-shadow regions
    rmse_shadow = torch.sqrt(torch.mean((gt_alpha[shadow_mask] - pred_alpha[shadow_mask])**2))
    rmse_nonshadow = torch.sqrt(torch.mean((gt_alpha[non_shadow_mask] - pred_alpha[non_shadow_mask])**2))

    # Compute the proportion of shadow region in gt_alpha
    ratio_gt = torch.sum(shadow_mask) / gt_alpha.numel()

    # Compute the Weighted Shadow Error
    wse = ((1 - ratio_gt) / ratio_gt) * rmse_shadow + rmse_nonshadow

    return wse
class MyConfuseMatrixMeter(AverageMeter):
    """More Clear Confusion Matrix Meter"""
    def __init__(self, n_class):
        super(MyConfuseMatrixMeter, self).__init__()
        self.n_class = n_class
        self.soft_rmse = None
        self.wse = None

    def update_cm(self, y_pred, y_label, soft_pred, self_gt, weight=1):
        y_label = y_label.type(torch.int64)
        val = compute_cm_torch(y_pred=y_pred.flatten(), y_label=y_label.flatten(),
                               n_class=self.n_class)
        self.update(val, weight)
        self.soft_rmse = compute_RMSE_torch(soft_pred, self_gt)
        self.wse = compute_wse_torch(soft_pred, self_gt)

    # def get_scores_binary(self):
    #     assert self.n_class == 2, "this function can only be called for binary calssification problem"
    #     tn, fp, fn, tp = self.sum.flatten()
    #     eps = torch.finfo(torch.float32).eps
    #     precision = tp / (tp + fp + eps)
    #     recall = tp / (tp + fn + eps)
    #     f1 = 2*recall*precision / (recall + precision + eps)
    #     iou = tp / (tp + fn + fp + eps)
    #     oa = (tp + tn) / (tp + tn + fn + fp + eps)
    #     score_dict = {}
    #     score_dict['precision'] = precision.item()
    #     score_dict['recall'] = recall.item()
    #     score_dict['f1'] = f1.item()
    #     score_dict['iou'] = iou.item()
    #     score_dict['oa'] = oa.item()
    #     return score_dict
    def get_scores_binary(self):
        assert self.n_class == 2, "this function can only be called for binary calssification problem"
        tn, fp, fn, tp = self.sum.flatten()
        eps = torch.finfo(torch.float32).eps
        pos_err = (1 - tp / (tp + fn + eps)) * 100
        neg_err = (1 - tn / (tn + fp + eps)) * 100
        ber = (pos_err + neg_err) / 2
        acc = (tn + tp) / (tn + tp + fn + fp + eps)
        score_dict = {}
        score_dict['pos_err'] = pos_err
        score_dict['neg_err'] = neg_err
        score_dict['ber'] = ber
        score_dict['acc'] = acc
        score_dict['soft_rmse'] = self.soft_rmse
        score_dict['wse'] = self.wse
        return score_dict
