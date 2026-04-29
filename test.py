import os
import torch
import cv2
import numpy as np

from models.wsl_net import WSLNet
from utils.cam import GradCAMPlusPlus
from config import Config

def post_process_mask(mask, kernel_size=5):
    """
    改进：使用形态学操作改进分割效果
    - 去除边缘噪点
    - 增强连通性
    - 平滑边界
    """
    # 创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 开运算（腐蚀后膨胀）：去除小的白点噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 闭运算（膨胀后腐蚀）：填补小孔洞，增强连通性
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return mask

def apply_intelligent_threshold(heatmap_uint8):
    """
    改进：智能二值化，避免 OTSU 的极端情况
    - 先尝试 OTSU
    - 如果效果不好，使用自适应阈值或固定阈值
    """
    if heatmap_uint8.max() == 0:
        return np.zeros_like(heatmap_uint8)
    
    # 首先尝试 OTSU 二值化
    _, mask_otsu = cv2.threshold(
        heatmap_uint8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # 检查分割结果是否合理（避免全白或全黑）
    white_ratio = mask_otsu.sum() / (256 * 256 * 255)
    
    # 如果白点过多（>50%）或过少（<1%），使用自适应阈值
    if white_ratio > 0.5 or white_ratio < 0.01:
        # 使用自适应高斯二值化，对于异常区域通常更稳定
        mask = cv2.adaptiveThreshold(
            heatmap_uint8, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # 必须是奇数
            C=2  # 常数，可根据需要调整
        )
        print(f"[使用自适应阈值] 白点比例: {white_ratio:.2%}")
    else:
        mask = mask_otsu
        print(f"[使用 OTSU 阈值] 白点比例: {white_ratio:.2%}")
    
    return mask


def test(img_path):
    device = Config.device

    os.makedirs(Config.cam_dir, exist_ok=True)
    os.makedirs(Config.mask_dir, exist_ok=True)

    model = WSLNet().to(device)
    model.load_state_dict(torch.load(Config.model_path))
    model.eval()

    cam = GradCAMPlusPlus(model)

    # ===== 读取图像 =====
    img_name = os.path.basename(img_path)

    img = cv2.imread(img_path)
    original_height, original_width = img.shape[:2]  # 保存原始尺寸
    img = cv2.resize(img, (256, 256))
    img_rgb = img[:, :, ::-1] / 255.0

    img_tensor = np.transpose(img_rgb, (2, 0, 1))
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).float().to(device)

    # ===== CAM =====
    heatmap = cam(img_tensor)  # 返回 [B, 1, H, W]
    heatmap = heatmap.squeeze().detach().cpu().numpy()  # [H, W]

    # ===== 数值统计调试信息 =====
    print(f"\n{'='*60}")
    print(f"图像: {img_name}")
    print(f"{'='*60}")
    print(f"CAM值范围: min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}")
    print(f"CAM非零像素: {(heatmap > 1e-6).sum()} / {heatmap.size} ({(heatmap > 1e-6).sum() / heatmap.size * 100:.2f}%)")
    print(f"CAM > 0.1的像素: {(heatmap > 0.1).sum()}")
    print(f"CAM > 0.5的像素: {(heatmap > 0.5).sum()}")

    # ===== 归一化处理 =====
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    else:
        print("[警告] CAM为常数，梯度可能未正确计算")
        heatmap = np.zeros_like(heatmap)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # ===== 恢复原始尺寸 =====
    heatmap_uint8 = cv2.resize(heatmap_uint8, (original_width, original_height), 
                                interpolation=cv2.INTER_LINEAR)

    # ===== 保存 CAM =====
    cam_path = os.path.join(Config.cam_dir, img_name)
    cv2.imwrite(cam_path, heatmap_uint8)
    print(f"✓ Saved CAM -> {cam_path}")

    # ===== 二值化 + 形态学操作 =====
    if heatmap_uint8.max() > 0:
        mask = apply_intelligent_threshold(heatmap_uint8)
        mask = post_process_mask(mask, kernel_size=5)
    else:
        mask = np.zeros_like(heatmap_uint8)
        print("[警告] CAM全零，mask也为全零")

    # ===== 恢复原始尺寸 =====
    mask = cv2.resize(mask, (original_width, original_height), 
                       interpolation=cv2.INTER_NEAREST)

    mask_path = os.path.join(Config.mask_dir, img_name)
    cv2.imwrite(mask_path, mask)
    print(f"✓ Saved MASK -> {mask_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test("test.jpg")
    test("test1.jpg")
    test("test2.jpg")
