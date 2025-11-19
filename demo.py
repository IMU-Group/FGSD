import os
import cv2
import torch
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
from networks.fdrnet import FDRNet

# 配置路径
ckpt_path = '/home/xuk/xuke/demos/SSD_2-main/ckpt/FSD_best.ckpt'
data_root = '/home/xuk/xuke/dataset/3D/chair_res800_var_cam_v1_sigma150'
# data_root =  '/home/xuk/xuke/dataset/FSD/test/test_A'
save_dir = '3D/chair/raw'
binary_dir = '3D/chair/binary'
soft_dir = '3D/chair/soft'

# 创建保存目录
os.makedirs(binary_dir, exist_ok=True)
os.makedirs(soft_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# 加载模型
model = FDRNet(backbone='efficientnet-b3',
               proj_planes=16,
               pred_planes=32,
               use_pretrained=True,
               fix_backbone=False,
               has_se=False,
               dropout_2d=0,
               normalize=True,
               mu_init=0.5,
               reweight_mode='manual')
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'])
model.cuda()
model.eval()

# 获取图像文件列表
image_files = [f for f in os.listdir(data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 推理并保存结果
with torch.no_grad():
    for image_file in tqdm(image_files):
        # 读取图像
        image_path = os.path.join(data_root, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = TF.to_tensor(image).unsqueeze(0).cuda()

        # 推理
        logit = model(image)['binary_mask']
        soft_mask = model(image)['soft_feat']
        pred_logit = F.interpolate(logit, size=image.size()[-2:], mode='bilinear')
        pred_soft = F.interpolate(soft_mask, size=image.size()[-2:], mode='bilinear').cpu()
        pred = (pred_logit > 0.5).type(torch.int64).cpu()

        # 保存结果
        save_path = os.path.join(save_dir, image_file)
        soft_path = os.path.join(save_dir.replace('raw', 'soft'), image_file)
        binary_path = os.path.join(save_dir.replace('raw', 'binary'), image_file)
        torchvision.utils.save_image([image.cpu()[0], pred[0].expand_as(image[0]), pred_soft[0].expand_as(image[0])], fp=save_path, nrow=3, padding=0)

        # 保存soft mask为8位灰度图
        soft_mask = (soft_mask[0] * 255).cpu()
        TF.to_pil_image(soft_mask[0].byte()).save(soft_path)

        # 保存预测的二值mask
        pred = (pred[0] * 255).cpu()
        pred = TF.to_pil_image(pred.byte())
        pred.save(binary_path)