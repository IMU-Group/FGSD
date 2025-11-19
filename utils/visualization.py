import torch
from torchvision.utils import draw_segmentation_masks
import torchvision

def colorize_classid_array(classid_array, image=None, alpha=0.8, colors=None):
    """
    Args:
        classidx_array: torch.LongTensor, (H, W) tensor
        num_cls: int, number of classes
        image: if None, overlay colored label on it, otherwise a pure black image is created
        colors: list/dict/array provdes class id to color mapping
    """
    if image is None:
        image = torch.zeros(size=(3, classid_array.size(-2), classid_array.size(-1)),
                            dtype=torch.uint8)
    # if colors is not None:
    #     assert len(colors) == num_cls, 'size of colormap should be consistent with num_cls'
    # all_class_masks = (classid_array == torch.arange(num_cls)[:, None, None])
    # im_label_overlay = draw_segmentation_masks(image, all_class_masks, alpha=alpha, colors=colors)
    unique_idx = torch.unique(classid_array)
    colors_use = [colors[idx.item()] for idx in unique_idx]
    all_class_masks = (classid_array == unique_idx[:, None, None])
    im_label_overlay = draw_segmentation_masks(image, all_class_masks, alpha=alpha, colors=colors_use)

    return im_label_overlay, unique_idx

def save_comparison_image(image, gt, pred, save_path):
    """
    Save a comparison image showing the ground truth (blue), prediction (red), and their overlap.
    
    Args:
        image (Tensor): The original image tensor of shape (C, H, W).
        gt (Tensor): The ground truth mask tensor of shape (H, W).
        pred (Tensor): The prediction mask tensor of shape (H, W).
        save_path (str): The path to save the comparison image.
    """
    # Ensure gt and pred are binary masks
    gt = gt > 0
    pred = pred > 0
    # 将gt和pred的mask转换为三通道，分别为蓝色和红色
    gt = gt.float()
    pred = pred.float()
    gt = gt.unsqueeze(0).repeat(3, 1, 1)
    pred = pred.unsqueeze(0).repeat(3, 1, 1)
    # 改变颜色
    gt[0] = gt[0] * 0
    gt[1] = gt[1] * 0
    gt[2] = gt[2] * 255
    pred[0] = pred[0] * 255
    pred[1] = pred[1] * 0
    pred[2] = pred[2] * 0
    # 将gt和pred的mask叠加到原图上
    overlay = gt + pred
    image = torch.where(overlay == 0, image, overlay)
    # 保存图片
    imgrid = torchvision.utils.save_image(overlay, fp=save_path)
    
