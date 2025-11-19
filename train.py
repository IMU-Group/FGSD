from sys import prefix
import torch
from torch.optim import lr_scheduler, optimizer
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torchvision.utils as vutils
from networks.fdrnet import FDRNet
from networks.loss import BBCEWithLogitLoss, DiceLoss
from utils.evaluation import MyConfuseMatrixMeter
import numpy as np
import random

from torch import Tensor
import torch.nn.functional as F

from datasets.sbu_dataset_new import SBUDataset
from datasets.transforms import Denormalize

from torch.utils.tensorboard import SummaryWriter

import configargparse
import os
import logging

from utils.visualization import colorize_classid_array
from networks.loss import alpha_loss,smooth_loss,RMSELoss,area_loss
import cv2
from PIL import Image
logger = logging.getLogger('ShadowDet')
logger.setLevel(logging.DEBUG)


def seed_all(seed=10):
    """
    https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    """
    logger.info(f"[ Using Seed : {seed} ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def create_logdir_and_save_config(args):
    paths = {}
    paths['sw_dir'] = os.path.join(args.logdir, 'summary')
    paths['ckpt_dir'] = os.path.join(args.logdir, 'ckpt')
    paths['val_dir'] = os.path.join(args.logdir, 'val')
    paths['test_dir'] = os.path.join(args.logdir, 'test')
    paths['log_file'] = os.path.join(args.logdir, 'train.log')
    paths['config_file'] = os.path.join(args.logdir, 'config.txt')
    paths['arg_file'] = os.path.join(args.logdir, 'args.txt')

    #### create directories #####
    for k, v in paths.items():
        if k.endswith('dir'):
            os.makedirs(v, exist_ok=True)

    #### create summary writer #####
    sw = SummaryWriter(log_dir=paths['sw_dir'])

    #### log to both console and file ####
    str2loglevel = {'info': logging.INFO, 'debug': logging.DEBUG, 'error': logging.ERROR}
    level = str2loglevel[args.loglevel]
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(paths['log_file'], 'w+')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #### print and save configs #####
    msg = 'Experiment arguments:\n ============begin==================\n'
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        msg += '{} = {}\n'.format(arg, attr)
    msg += '=============end================'
    logger.info(msg)
    
    with open(paths['arg_file'], 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        with open(paths['config_file'], 'w') as file:
            file.write(open(args.config, 'r').read())

    return sw, paths


def create_model_and_optimizer(args):
    """
    return model, optimizer, lr_schedule, start_epoch
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # create model
    model = FDRNet(backbone='efficientnet-b3',
               proj_planes=16,
               pred_planes=32,
               use_pretrained=True,
               fix_backbone=False,
               has_se=False,
               dropout_2d=0,
               normalize=True,
               mu_init=args.mu_init,
               reweight_mode='manual')

    model.cuda(0)
    
    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    # lr schedule
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    start_epoch = 0

    logger.info(f'model {args.model} is created!')

    if args.ckpt is not None:
        model, optimizer, lr_schedule, start_epoch = load_ckpt(model, optimizer, lr_schedule, args.ckpt)

    return model, optimizer, lr_schedule, start_epoch


def create_loss_function(args):
    if args.loss == 'bce':
        loss_function = nn.BCEWithLogitsLoss()
    elif args.loss == 'dice':
        loss_function = DiceLoss()
    elif args.loss == 'bbce':
        loss_function = BBCEWithLogitLoss()
    else:
        raise ValueError(f'{args.loss} is not supported!')

    return loss_function


def create_dataloaders(args):
    data_roots = {'SBU_train': '/home/xuk/xuke/dataset/SBU_soft/train',
                  'SBU_test': '/home/xuk/xuke/dataset/SBU_soft/test',
                  'UCF_test': './local_data/UCF/GouSplit',
                  'ISTD_train': '/home/xuk/xuke/dataset/ISTD_plus/train',
                  'ISTD_test': '/home/xuk/xuke/dataset/ISTD_plus/test',
                  'ISS2K_train': '/home/xuk/xuke/dataset/ISS2K_edited/train',
                  'ISS2K_test': '/home/xuk/xuke/dataset/ISS2K_edited/test',
                  'ISTD_plus_train': '/home/xuk/xuke/dataset/ISTD_plus/train',
                  'ISTD_plus_test': '/home/xuk/xuke/dataset/ISTD_plus/test',
                  'SRD_train': '/home/xuk/xuke/dataset/SRD/train',
                  'SRD_test': '/home/xuk/xuke/dataset/SRD/test',
                  'SRD_plus_train': '/home/xuk/xuke/dataset/SRD_plus2/train',
                  'SRD_plus_test': '/home/xuk/xuke/dataset/SRD_plus2/test',
                  'FSD_train': '/home/xuk/xuke/dataset/FSD/train',
                  'FSD_test': '/home/xuk/xuke/dataset/FSD/test',}

    train_dataset = SBUDataset(data_root=data_roots[args.train_data],
                               phase='train', augmentation=False, im_size=args.train_size, normalize=False,img_dirs=['train_A'],mask_dir='train_B',soft_mask_dir='train_E')
    
    ## set drop_last True to avoid error induced by BatchNormalization
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch,
                                               shuffle=True, num_workers=args.nworker,
                                               pin_memory=True, drop_last=True)
    
    # name, split = args.eval_data.split('_')
    # val_dataset = get_datasets(name=name,
    #                            root=os.path.join(args.data_root, name),
    #                            split=split,
    #                            transform=val_tf
    #                           )
    eval_data = args.eval_data.split('+')
    eval_loaders = {}
    for name in eval_data:
        dataset = SBUDataset(data_root=data_roots[name], phase='test', augmentation=False,
                             im_size=args.eval_size, normalize=False,img_dirs=['test_A'],mask_dir='test_B',soft_mask_dir='test_E')
        eval_loaders[name] = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch,
                                            shuffle=False, num_workers=args.nworker,
                                            pin_memory=True)
    
    # msg = "Dataloaders are prepared!\n=============================\n"
    # msg += f"train_loader: dataset={args.train_data}, num_samples={len(train_loader.dataset)}, batch_size={train_loader.batch_size}\n"
    # msg += f"val_loader: dataset={args.eval_data}, num_samples={len(val_loader.dataset)}, batch_size={val_loader.batch_size}\n"
    # # msg += f"test_loader: dataset={args.test_data}, num_samples={len(test_loader.dataset)}, batch_size={test_loader.batch_size}\n"
    # # msg += "------------------------------\n"
    # # msg += f"load_size={args.load_size}\n"
    # msg += "============================="
    msg = "Dataloaders are prepared!"
    logger.info(msg)
    
    return train_loader, eval_loaders


def save_ckpt(model, optimizer, lr_schedule, epoch, path):
    ckpt = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': lr_schedule.state_dict(),
            'epoch': epoch
           }
    torch.save(ckpt, path)
    logger.info(f'checkpoint has been saved to {path}!')


def load_ckpt(model, optimizer, lr_schedule, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    lr_schedule.load_state_dict(ckpt['lr_schedule'])
    start_epoch = ckpt['epoch'] + 1
    logger.info(f'model is loaded from {path}!')
    return model, optimizer, lr_schedule, start_epoch


def visualize_sample(images: Tensor, gt: Tensor, pred: Tensor, bi_th=0.5):
    """
    visualize single sample
    Args:
        images: [2, 3, h, w] tensor
        gt: [1, h, w] tensor, int binary mask
        pred: [1, h, w] tensor, float soft mask
    Return:
        grid: visual grid
    """
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    # # mean=[0.5, 0.5, 0.5]
    # # std=[0.5, 0.5, 0.5]
    # denorm_fn = Denormalize(mean=mean, std=std)
    # images_vis = (denorm_fn(images)*255).type(torch.uint8).cpu()
    images_vis = (images*255).type(torch.uint8).cpu()
    gt_vis = (torch.cat([gt*255]*3, dim=0)).type(torch.uint8).cpu()
    pred_vis = (torch.cat([pred*255]*3, dim=0)).type(torch.uint8).cpu()
    pred_bi_vis = (torch.cat([(pred>bi_th).float()*255]*3, dim=0)).type(torch.uint8).cpu()

    # -1: false negative, 0: correct, 1: false positive
    diff = (pred > bi_th).type(torch.int8) - gt.type(torch.int8)
    logger.debug(f"unique_ids in pred_bi_vis: {torch.unique(pred_bi_vis)}")
    logger.debug(f"unique_ids in diff: {torch.unique(diff)}")
    diff_vis, _ = colorize_classid_array(diff, alpha=1., image=None,
                        colors={-1: 'green', 0:'black', 1:'red'})
    diff_vis = diff_vis.cpu()
    grid = vutils.make_grid([images_vis, gt_vis, pred_vis, pred_bi_vis, diff_vis],
                             nrow=3, padding=0)
    return grid


@torch.no_grad()
def evaluate(model, eval_loader, bi_class_th=0.5, save_dir=None, sw=None, epoch=None, prefix=''):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    model.eval()
    cmm = MyConfuseMatrixMeter(n_class=2)
    for i_batch, data in enumerate(tqdm(eval_loader)):
        inp = data['ShadowImages_input'].cuda(0)
        gt = data['gt'].cuda(0)
        gt_soft = data['soft_gt'].cuda(0)
        logit = model(inp)['binary_mask']
        soft_mask = model(inp)['soft_feat']
        pred_logit = F.interpolate(logit, size=gt.size()[-2:], mode='bilinear')
        pred_soft = F.interpolate(soft_mask, size=gt.size()[-2:], mode='bilinear')
        pred = (pred_logit > bi_class_th).type(torch.int64)
        concat_pred = torch.cat([gt[0], pred[0]], dim=2)
        sw.add_image('concat_pred', concat_pred, global_step=epoch)
        sw.add_image('pred_soft_mask', pred_soft[0], global_step=epoch)
        cmm.update_cm(y_pred=pred.cpu(), y_label=gt.cpu(), soft_pred=pred_soft.cpu(), self_gt=gt_soft.cpu())
        if save_dir is not None:
            inp = F.interpolate(inp, size=gt.size()[-2:], mode='bilinear')
            for i_image, (x, y_gt, y_pred) in enumerate(zip(inp, gt, pred_soft)):
                im_grid = visualize_sample(images=x, gt=y_gt, pred=y_pred)
                save_name = f'{i_batch*eval_loader.batch_size + i_image:05d}.png'
                save_path = os.path.join(save_dir, save_name)
                vutils.save_image(im_grid/255., save_path)
    score_dict = cmm.get_scores_binary()
    msg = 'Scores:\n==============================================='
    for k, v in score_dict.items():
        msg += f'\n\t{prefix}.{k}:{v}'
        if sw is not None:
            sw.add_scalar(f'eval/{prefix}.{k}', v, global_step=epoch)
    msg += '\n==============================================='
    logger.info(msg)
    return score_dict


def train(model, train_loader, loss_fn, optimizer, lr_schedule, epoch, sw, args):
    """
    one epoch scan
    """
    global_step = epoch * len(train_loader)
    model.train()
    for i_batch, sample in enumerate(train_loader): # mini-batch update
        global_step += 1
        image, label, gt_alpha = sample['ShadowImages_input'].cuda(0), sample['gt'].cuda(0), sample['soft_gt'].cuda(0)
        logit, f_inv, delta_pred_1, soft_mask,binary_mask = model(image)
        loss_det = loss_fn(logit, label)
        loss_alpha = alpha_loss(soft_mask, gt_alpha)
        loss_smooth = smooth_loss(soft_mask, gt_alpha)
        loss_rmse = RMSELoss(soft_mask, gt_alpha)
        delta = np.random.uniform(-0.3, 0.3)
        delta_matix = torch.ones_like(image) * delta
        delta_matix = delta_matix * label
        view2 = torch.clamp(image + delta_matix, 0., 1.)
        binary_loss = loss_fn(binary_mask, label)
        
        _, f_inv_view2, delta_pred, _soft_mask,__ = model(view2)
        loss_inv = F.l1_loss(f_inv_view2, f_inv.detach())
        delta_pred = delta_pred + delta_pred_1
        loss_var = F.l1_loss(delta_pred, delta * torch.ones_like(delta_pred))
        loss_total = args.loss_inv_coeff * loss_inv + args.loss_var_coeff * loss_var + args.loss_rmse_coeff * loss_rmse + args.loss_det_coeff * loss_det + args.loss_smooth_coeff * loss_smooth + args.loss_binary_coeff * binary_loss
        # loss_total = 1.0 * loss_inv + 1.0 * loss_var + 1.0 * loss_rmse + 10 * loss_det + 1.0 * loss_alpha + 10 * loss_smooth + 10 * binary_loss
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        if global_step % args.i_print == 0 and i_batch > 0:
            info_dict = {'loss_rmse': loss_rmse.item(), 'loss_inv': loss_inv.item(), 'loss_var': loss_var.item(), "loss_det": loss_det.item(), "loss_alpha": loss_alpha.item(), "loss_smooth": loss_smooth.item(), "loss_total": loss_total.item(), "binary_loss": binary_loss.item()}
            msg = f'[batch {i_batch+1}/{len(train_loader)}, epoch {epoch+1}/{args.total_ep}]: '
            for k, v in info_dict.items():
                msg += f'{k}:{v} '
                sw.add_scalar(f'train/{k}', v, global_step=global_step)
            logger.info(msg)
    lr_schedule.step()
    model.fr.set_mu(1 - (epoch/args.total_ep)**2)


def main(args):
    seed_all(args.seed)
    sw, paths = create_logdir_and_save_config(args)
    model, optimizer, lr_schedule, start_epoch = create_model_and_optimizer(args)
    loss_fn = create_loss_function(args)
    train_loader, val_loader_dict = create_dataloaders(args)
    best_ber = float('inf')
    best_epoch = -1
    if args.action == 'test':
        for name, val_loader in val_loader_dict.items():
            _ = evaluate(model, val_loader, bi_class_th=args.prob_th, save_dir=os.path.join(paths['test_dir'], name), prefix=name)
    elif args.action == 'train':
        for epoch in range(start_epoch, args.total_ep):
            train(model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, lr_schedule=lr_schedule, epoch=epoch, sw=sw, args=args)
            for name, val_loader in val_loader_dict.items():
                score_dict = evaluate(model, val_loader, bi_class_th=args.prob_th, save_dir=None, sw=sw, epoch=epoch, prefix=name)
                if score_dict['ber'] < best_ber:
                    best_ber = score_dict['ber']
                    best_epoch = epoch
                    ckpt_path = os.path.join(paths['ckpt_dir'], 'best.ckpt')
                    save_ckpt(model, optimizer, lr_schedule, epoch, path=ckpt_path)
            # if args.save_ckpt > 0:
            #     ckpt_path = os.path.join(paths['ckpt_dir'], f'ep_{epoch:03d}.ckpt')
            #     save_ckpt(model, optimizer, lr_schedule, epoch, path=ckpt_path)
        logger.info(f'Best BER: {best_ber:.4f} at epoch {best_epoch}')
    else:
        raise ValueError(f'invalid action {args.action}')
    sw.close()
    hdls = logger.handlers[:]
    for handler in hdls:
        handler.close()
        logger.removeHandler(handler)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    
    parser.add_argument('--action', type=str, default='train', choices=['train', 'test'],
                        help='action, train or test')

    ## model
    parser.add_argument('--model', type=str, default='BANet.efficientnet-b3', 
                        help='architecture to be used')
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt to load')

    ## optimization
    parser.add_argument('--seed', type=int, default=1, help='random seed.')
    parser.add_argument('--total_ep', type=int, default=15,  help='number of epochs for training.')
    parser.add_argument('--lr', type=float, default=5e-4,  help='initial learning rate.')
    parser.add_argument('--lr_step', type=int, default=1,  help='learning rate decay frequency (in epochs).')
    parser.add_argument('--lr_gamma', type=float, default=0.7,  help='learning rate decay factor.')
    parser.add_argument('--wd', type=float, default=1e-4,  help='weight decay.')
    parser.add_argument('--loss', type=str, default='bbce', help='loss function')
    parser.add_argument('--save_ckpt', type=int, default=1, help='>0 means save ckpt during training.')

    ## loss coefficients for ablation study
    parser.add_argument('--loss_inv_coeff', type=float, default=0.0, help='coefficient for invariance loss')
    parser.add_argument('--loss_var_coeff', type=float, default=1.0, help='coefficient for variance loss')
    parser.add_argument('--loss_rmse_coeff', type=float, default=1.0, help='coefficient for RMSE loss')
    parser.add_argument('--loss_det_coeff', type=float, default=10.0, help='coefficient for detection loss')
    parser.add_argument('--loss_smooth_coeff', type=float, default=10.0, help='coefficient for smooth loss')
    parser.add_argument('--loss_binary_coeff', type=float, default=10.0, help='coefficient for binary loss')

    ## model parameters
    parser.add_argument('--mu_init', type=float, default=0.5, help='initial value for mu parameter in FDRNet')

    ## data
    parser.add_argument('--train_data', type=str, default='FSD_train', help='training dataset')
    parser.add_argument('--eval_data', type=str, default='FSD_test', help='eval dataset')
    parser.add_argument('--train_batch', type=int, default=8, help='batch_size for train and val dataloader.')
    parser.add_argument('--eval_batch', type=int, default=1, help='batch_size for train and val dataloader.')
    parser.add_argument('--train_size', type=int, default=512, help='scale images to this size for training')
    parser.add_argument('--eval_size', type=int, default=512, help='scale images to this size for evaluation')
    parser.add_argument('--nworker', type=int, default=2, help='num_workers for train and val dataloader.')

    ## evaluation
    parser.add_argument('--prob_th', type=float, default=0.5,  help='threshold for binary classification.')

    ## logging
    parser.add_argument('--logdir', type=str, default='logs', help='directory to save logs, args, etc.')
    parser.add_argument('--loglevel', type=str, default='info', help='logging level.')
    parser.add_argument('--i_print', type=int, default=10, help='training loss display frequency in mini-batchs.')
    # parser.add_argument('--i_vis', type=int, default=100, help='training loss display frequency in steps.')

    # ## test
    # parser.add_argument('--test_out', type=str, default='local_test_out', help='directory to save test visuals.')
    
    args = parser.parse()
    main(args) 