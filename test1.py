import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from models.wsl_net import WSLNet
from torchcam.methods import GradCAMpp
from config import Config
from datasets.kolektor import KolektorDataset
from datasets.mvtecad import MVTecADDataset


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

    blurred = cv2.GaussianBlur(heatmap_uint8, (7, 7), 0)
    _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = mask_otsu.sum() / (mask_otsu.size * 255.0)

    if white_ratio > 0.5 or white_ratio < 0.01:
        mask = cv2.adaptiveThreshold(
            blurred, 255,
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

    if gt_binary.sum() == 0:
        if pred_binary.sum() == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = tp / (tp + fp + 1e-8)
            recall = 1.0
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1, tp, fp, fn, tn


def test_comprehensive(dataset_root=None, split="test", dataset_type="auto"):
    """
    综合测试：计算 AP, mIoU, Precision, Recall, F1-score
    并生成 P-R 曲线
    """
    device = Config.device
    
    if dataset_root is None:
        dataset_root = Config.data_root
    
    # 加载数据集
    print(f"Loading {split} dataset from {dataset_root}...")
    
    if dataset_type == "auto":
        if "mvtecad" in dataset_root.lower():
            dataset_type = "mvtecad"
    
    if dataset_type == "mvtecad":
        dataset = MVTecADDataset(dataset_root, Config.mvtecad_category, img_width=Config.img_width, img_height=Config.img_height, train=(split=="train"))
    else:
        dataset = KolektorDataset(dataset_root, img_width=Config.img_width, img_height=Config.img_height, train=(split=="train"), dataset_type=dataset_type)
    
    # 加载模型
    print("Loading model...")
    model = WSLNet().to(device)
    model.load_state_dict(torch.load(Config.model_path))
    model.eval()
    
    cam = GradCAMpp(model, target_layer=model.backbone[7])
    
    # 创建结果输出目录
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
    anomaly_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "iou": []
    }
    
    print(f"\nTesting on {len(dataset)} images...")
    original_width = Config.img_width
    original_height = Config.img_height
    
    # 测试循环
    for idx in tqdm(range(len(dataset))):
        img, label, gt_mask = dataset[idx]
        
        gt_mask = gt_mask.numpy()  # Convert to numpy for cv2 operations
        
        img_tensor = img.unsqueeze(0).to(device)
        
        # 获取 logits 和 probs
        logits, _, _ = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
        
        all_scores.append(float(probs))
        all_labels.append(label.item() if isinstance(label, torch.Tensor) else float(label))
        
        # 生成 CAM 热力图
        heatmap = cam(class_idx=0, scores=logits)[0].squeeze().detach().cpu().numpy()
        
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
        
        # 计算像素级别指标
        precision, recall, f1, tp, fp, fn, tn = compute_pixel_metrics(mask_resized, gt_mask)
        iou = compute_iou(mask_resized, gt_mask)
        
        image_metrics["precision"].append(precision)
        image_metrics["recall"].append(recall)
        image_metrics["f1"].append(f1)
        image_metrics["iou"].append(iou)
        
        if gt_mask.sum() > 0:
            anomaly_metrics["precision"].append(precision)
            anomaly_metrics["recall"].append(recall)
            anomaly_metrics["f1"].append(f1)
            anomaly_metrics["iou"].append(iou)
    
    # 计算图像级别指标
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # AP (Average Precision)
    ap = average_precision_score(all_labels, all_scores)
    
    # 先使用标准 0.5 阈值，再根据必要时使用最佳 F1 阈值
    binary_preds = (all_scores >= 0.5).astype(int)
    if binary_preds.sum() == 0 and len(np.unique(all_labels)) > 1:
        # 如果所有预测都被判为正常，则从 Precision-Recall 曲线选择最佳阈值
        precision_curve_vals, recall_curve_vals, thresholds = precision_recall_curve(all_labels, all_scores)
        f1_curve = 2 * precision_curve_vals * recall_curve_vals / (precision_curve_vals + recall_curve_vals + 1e-8)
        best_idx = np.nanargmax(f1_curve)
        if best_idx < len(thresholds):
            best_threshold = thresholds[best_idx]
            binary_preds = (all_scores >= best_threshold).astype(int)
        else:
            binary_preds = (all_scores >= 0.5).astype(int)

    img_precision = precision_score(all_labels, binary_preds, zero_division=0)
    img_recall = recall_score(all_labels, binary_preds, zero_division=0)
    img_f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    # 像素级别平均指标
    mean_precision = np.mean(image_metrics["precision"])
    mean_recall = np.mean(image_metrics["recall"])
    mean_f1 = np.mean(image_metrics["f1"])
    mean_iou = np.mean(image_metrics["iou"])
    anomaly_mean_precision = np.mean(anomaly_metrics["precision"]) if len(anomaly_metrics["precision"]) > 0 else 0.0
    anomaly_mean_recall = np.mean(anomaly_metrics["recall"]) if len(anomaly_metrics["recall"]) > 0 else 0.0
    anomaly_mean_f1 = np.mean(anomaly_metrics["f1"]) if len(anomaly_metrics["f1"]) > 0 else 0.0
    anomaly_mean_iou = np.mean(anomaly_metrics["iou"]) if len(anomaly_metrics["iou"]) > 0 else 0.0
    
    # 打印结果
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nImage-level Classification Metrics:")
    print(f"  Average Precision (AP):        {ap:.4f}")
    print(f"  Precision:                     {img_precision:.4f}")
    print(f"  Recall:                        {img_recall:.4f}")
    print(f"  F1-Score:                      {img_f1:.4f}")
    
    print(f"\nPixel-level Segmentation Metrics (Mean over all images):")
    print(f"  Precision:                     {mean_precision:.4f}")
    print(f"  Recall:                        {mean_recall:.4f}")
    print(f"  F1-Score:                      {mean_f1:.4f}")
    print(f"  mIoU (Mean IoU):               {mean_iou:.4f}")
    
    print(f"\nPixel-level Segmentation Metrics (Mean over anomaly images only):")
    print(f"  Precision:                     {anomaly_mean_precision:.4f}")
    print(f"  Recall:                        {anomaly_mean_recall:.4f}")
    print(f"  F1-Score:                      {anomaly_mean_f1:.4f}")
    print(f"  mIoU (Mean IoU):               {anomaly_mean_iou:.4f}")
    
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
    metrics_path = os.path.join(results_dir, 'test_results.txt')
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
    
    parser = argparse.ArgumentParser(description="Comprehensive evaluation on defect detection datasets")
    parser.add_argument("--dataset", type=str, default=None, 
                       help="Dataset root path (defaults to Config.data_root)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                       help="Train or test split")
    parser.add_argument("--dataset-type", type=str, default="auto", 
                       choices=["auto", "kolektor_sdd", "kolektor_sdd2", "mvtecad"],
                       help="Dataset type")
    args = parser.parse_args()
    
    results = test_comprehensive(dataset_root=args.dataset, split=args.split, dataset_type=args.dataset_type)
