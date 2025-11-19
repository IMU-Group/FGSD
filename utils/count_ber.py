import os
import cv2
import numpy as np
import random
def calculate_ber(mask_pred, mask_gt):
    """计算Balanced Error Rate (BER)"""
    mask_pred = (mask_pred > 0).astype(np.uint8)
    mask_gt = (mask_gt > 0).astype(np.uint8)
    
    tp = np.sum((mask_pred == 1) & (mask_gt == 1))
    fp = np.sum((mask_pred == 1) & (mask_gt == 0))
    tn = np.sum((mask_pred == 0) & (mask_gt == 0))
    fn = np.sum((mask_pred == 0) & (mask_gt == 1))
    
    ber = 0
    denominator1 = tp + fn
    denominator2 = tn + fp
    if denominator1 > 0 and denominator2 > 0:
        ber = (fn / denominator1 + fp / denominator2) / 2
    return ber * 100

def create_error_map(pred, gt):
    """创建错误检测可视化图"""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    error_map = np.zeros((*pred.shape, 3), dtype=np.uint8)
    error_map[(pred == 1) & (gt == 0)] = [0, 0, 255]   # FP红色
    error_map[(pred == 0) & (gt == 1)] = [255, 0, 0]   # FN蓝色
    error_map[(pred == 1) & (gt == 1)] = [0, 255, 0]   # TP绿色
    return error_map

def process_folders(pred_dir, gt_dir, image_dir, output_dir, ber_threshold=5):
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化记录文件（清空已有内容）
    record_path = os.path.join(output_dir, "high_ber_files.txt")
    with open(record_path, "w") as f:  # 使用"w"模式清空文件
        pass
    
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

    for pred_file, gt_file, image_file in zip(pred_files, gt_files, image_files):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)
        image_path = os.path.join(image_dir, image_file)
        
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path)
        
        # 检查是否成功加载文件
        if pred is None or gt is None or image is None:
            print(f"跳过文件：{pred_file}, {gt_file}, {image_file}，因为无法加载")
            continue
        
        
        ber = calculate_ber(pred, gt)
        
        if ber > ber_threshold and random.random() < 0.7:  # 随机选择50%的高BER文件
            # 记录文件名到txt
            with open(record_path, "a") as f:  # 使用"a"模式追加
                f.write(f"{pred_file}\n")
            
            # 图像处理流程
            h, w = pred.shape
            gt_resized = cv2.resize(gt, (w, h))
            image_resized = cv2.resize(image, (w, h))
            
            pred_color = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            gt_color = cv2.cvtColor(gt_resized, cv2.COLOR_GRAY2BGR)
            error_map = create_error_map(pred, gt_resized)
            
            # 添加文字标注
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_resized, 'Image', (10,30), font, 0.7, (0,255,0), 2)
            cv2.putText(gt_color, 'GT', (10,30), font, 0.7, (0,255,0), 2)
            cv2.putText(pred_color, f'Pred (BER:{ber:.2f}%)', (10,30), font, 0.7, (0,255,0), 2)
            cv2.putText(error_map, 'Error Map', (10,30), font, 0.7, (255,255,255), 2)
            
            # 四图拼接
            combined = np.hstack((image_resized, gt_color, pred_color, error_map))
            output_path = os.path.join(output_dir, f"combined_{pred_file}")
            cv2.imwrite(output_path, combined)

if __name__ == "__main__":
    # 输入文件夹路径
    pred_dir = "/home/xuk/xuke/demos/SSD_2-main/test_FSD/binary"    # 预测mask文件夹
    gt_dir = "/home/xuk/xuke/dataset/FSD/test2/test_B"        # 真实mask文件夹
    image_dir = "/home/xuk/xuke/dataset/FSD/test2/test_A"  # 原始图像文件夹
    output_dir = "/home/xuk/xuke/demos/SSD_2-main/test_FSD/untrust" # 输出文件夹
    
    # 处理文件夹
    process_folders(pred_dir, gt_dir, image_dir, output_dir, ber_threshold=5)