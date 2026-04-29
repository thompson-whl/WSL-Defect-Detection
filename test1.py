import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from models.wsl_net import WSLNet
from utils.cam import GradCAMPlusPlus
from config import Config
from datasets.kolektor import KolektorDataset


def post_process_mask(mask, kernel_size=5):
    """
    改进：使用形态学操作改进分割效果
    - 去除边缘噪点
    - 增强连通性
    - 平滑边界
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def apply_intelligent_threshold(heatmap_uint8):
    """
    改进：智能二值化，避免 OTSU 的极端情况
    """
    if heatmap_uint8.max() == 0:
        return np.zeros_like(heatmap_uint8)
    
    _, mask_otsu = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = mask_otsu.sum() / (256 * 256 * 255)
    
    if white_ratio > 0.5 or white_ratio < 0.01:
        mask = cv2.adaptiveThreshold(
            heatmap_uint8, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
    else:
        mask = mask_otsu
    
    return mask


def compute_iou(pred_mask, gt_mask):
    """计算 IoU (Intersection over Union)"""
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_and(pred_binary, gt_binary).sum() + np.logical_xor(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_pixel_metrics(pred_mask, gt_mask):
    """计算像素级别的 precision, recall, F1"""
    pred_binary = (pred_mask > 127).astype(np.uint8).flatten()
    gt_binary = (gt_mask > 0).astype(np.uint8).flatten()
    
    tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
    fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
    fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
    tn = np.logical_and(pred_binary == 0, gt_binary == 0).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1, tp, fp, fn, tn


def test_comprehensive(dataset_root="./KolektorSDD2", split="test"):
    """
    综合测试：计算 AP, mIoU, Precision, Recall, F1-score
    并生成 P-R 曲线
    """
    device = Config.device
    
    # 加载数据集
    print(f"Loading {split} dataset from {dataset_root}...")
    dataset = KolektorDataset(dataset_root, img_size=Config.img_size, train=(split=="train"), 
                             dataset_type="kolektor_sdd2")
    
    # 加载模型
    print("Loading model...")
    model = WSLNet().to(device)
    model.load_state_dict(torch.load(Config.model_path))
    model.eval()
    
    cam = GradCAMPlusPlus(model)
    
    # 创建输出目录
    os.makedirs(Config.cam_dir, exist_ok=True)
    os.makedirs(Config.mask_dir, exist_ok=True)
    results_dir = os.path.join(Config.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化指标存储
    all_scores = []  # 图像级别的预测分数
    all_labels = []  # 图像级别的真实标签
    
    image_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "iou": []
    }
    
    print(f"\nTesting on {len(dataset)} images...")
    
    # 测试循环
    for idx in tqdm(range(len(dataset))):
        img_path = dataset.imgs[idx]
        label = dataset.labels[idx]
        label_map_path = dataset.label_maps[idx]
        
        # 读取图像
        img = cv2.imread(img_path)
        original_height, original_width = img.shape[:2]
        img_resized = cv2.resize(img, (256, 256))
        img_rgb = img_resized[:, :, ::-1] / 255.0
        
        img_tensor = np.transpose(img_rgb, (2, 0, 1))
        img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)
        
        # 获取 CAM
        with torch.no_grad():
            logits, _, _ = model(img_tensor)
            logits_np = logits.squeeze().cpu().numpy()
        
        all_scores.append(logits_np)
        all_labels.append(label)
        
        # 生成 CAM 热力图
        heatmap = cam(img_tensor)
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        
        # 归一化
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)
        
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap_uint8, (original_width, original_height), 
                                      interpolation=cv2.INTER_LINEAR)
        
        # 生成掩膜
        if heatmap_uint8.max() > 0:
            mask = apply_intelligent_threshold(heatmap_uint8)
            mask = post_process_mask(mask, kernel_size=5)
        else:
            mask = np.zeros_like(heatmap_uint8)
        
        mask_resized = cv2.resize(mask, (original_width, original_height), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # 读取真实标签
        gt_label = cv2.imread(label_map_path, cv2.IMREAD_GRAYSCALE)
        if gt_label is None:
            continue
        gt_label_resized = cv2.resize(gt_label, (original_width, original_height),
                                      interpolation=cv2.INTER_NEAREST)
        
        # 计算像素级别指标
        precision, recall, f1, tp, fp, fn, tn = compute_pixel_metrics(mask_resized, gt_label_resized)
        iou = compute_iou(mask_resized, gt_label_resized)
        
        image_metrics["precision"].append(precision)
        image_metrics["recall"].append(recall)
        image_metrics["f1"].append(f1)
        image_metrics["iou"].append(iou)
        
        # 保存结果
        img_name = os.path.basename(img_path)
        cam_path = os.path.join(Config.cam_dir, img_name)
        cv2.imwrite(cam_path, heatmap_resized)
        
        mask_path = os.path.join(Config.mask_dir, img_name)
        cv2.imwrite(mask_path, mask_resized)
    
    # 计算图像级别指标
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # AP (Average Precision)
    ap = average_precision_score(all_labels, all_scores)
    
    # 图像级别分类指标
    binary_preds = (all_scores > 0.5).astype(int)
    img_precision = precision_score(all_labels, binary_preds, zero_division=0)
    img_recall = recall_score(all_labels, binary_preds, zero_division=0)
    img_f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    # 像素级别平均指标
    mean_precision = np.mean(image_metrics["precision"])
    mean_recall = np.mean(image_metrics["recall"])
    mean_f1 = np.mean(image_metrics["f1"])
    mean_iou = np.mean(image_metrics["iou"])
    
    # 打印结果
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nImage-level Classification Metrics:")
    print(f"  Average Precision (AP):        {ap:.4f}")
    print(f"  Precision:                     {img_precision:.4f}")
    print(f"  Recall:                        {img_recall:.4f}")
    print(f"  F1-Score:                      {img_f1:.4f}")
    
    print(f"\nPixel-level Segmentation Metrics (Mean):")
    print(f"  Precision:                     {mean_precision:.4f}")
    print(f"  Recall:                        {mean_recall:.4f}")
    print(f"  F1-Score:                      {mean_f1:.4f}")
    print(f"  mIoU (Mean IoU):               {mean_iou:.4f}")
    
    print("="*70 + "\n")
    
    # 生成 P-R 曲线
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # P-R curve
    axes[0].plot(recall_curve, precision_curve, 'b-', linewidth=2)
    axes[0].fill_between(recall_curve, precision_curve, alpha=0.2)
    axes[0].set_xlabel('Recall', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title(f'Precision-Recall Curve (AP={ap:.4f})', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # 分布直方图
    axes[1].hist(all_scores[all_labels == 0], bins=30, alpha=0.6, label='Normal', color='green')
    axes[1].hist(all_scores[all_labels == 1], bins=30, alpha=0.6, label='Anomaly', color='red')
    axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    axes[1].set_xlabel('Model Output Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Score Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    pr_curve_path = os.path.join(results_dir, 'pr_curve.png')
    plt.savefig(pr_curve_path, dpi=150, bbox_inches='tight')
    print(f"✓ P-R curve saved to {pr_curve_path}")
    
    # 保存指标到文件
    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"\nImage-level Classification Metrics:\n")
        f.write(f"  Average Precision (AP):        {ap:.4f}\n")
        f.write(f"  Precision:                     {img_precision:.4f}\n")
        f.write(f"  Recall:                        {img_recall:.4f}\n")
        f.write(f"  F1-Score:                      {img_f1:.4f}\n")
        f.write(f"\nPixel-level Segmentation Metrics (Mean):\n")
        f.write(f"  Precision:                     {mean_precision:.4f}\n")
        f.write(f"  Recall:                        {mean_recall:.4f}\n")
        f.write(f"  F1-Score:                      {mean_f1:.4f}\n")
        f.write(f"  mIoU (Mean IoU):               {mean_iou:.4f}\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Metrics saved to {metrics_path}\n")
    
    return {
        "ap": ap,
        "img_precision": img_precision,
        "img_recall": img_recall,
        "img_f1": img_f1,
        "pixel_precision": mean_precision,
        "pixel_recall": mean_recall,
        "pixel_f1": mean_f1,
        "miou": mean_iou
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive evaluation on KolektorSDD2")
    parser.add_argument("--dataset", type=str, default="./KolektorSDD2", 
                       help="KolektorSDD2 root path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                       help="Train or test split")
    args = parser.parse_args()
    
    results = test_comprehensive(dataset_root=args.dataset, split=args.split)
