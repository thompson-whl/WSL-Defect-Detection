import torch
import torch.nn.functional as F

class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.features = None
        self.handles = []

    def save_grad(self, grad):
        """保存梯度"""
        self.gradients = grad

    def save_feat(self, feat):
        """保存特征图"""
        self.features = feat

    def register_hooks(self, layer):
        """为指定层注册钩子"""
        def forward_hook(module, input, output):
            self.features = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = layer.register_forward_hook(forward_hook)
        h2 = layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h1, h2])

    def remove_hooks(self):
        """移除所有钩子"""
        for h in self.handles:
            h.remove()
        self.handles = []

    def __call__(self, x):
        # ✅ 改进：正确的梯度计算设置
        self.model.eval()
        
        # ✅ 重要：设置模型为不需要进一步梯度的评估模式，但保留梯度计算
        # 这样 BatchNorm 使用运行统计而不是批统计，但仍然计算梯度
        
        # 为IAM后的特征注册钩子（获取最后的特征图）
        self.register_hooks(self.model.iam)

        with torch.enable_grad():
            # ✅ 修复：正确的输入准备
            x = x.clone().detach().requires_grad_(True)
            
            # ✅ 确保输入在正确的设备和数据类型上
            x = x.to(next(self.model.parameters()).device)
            
            # ✅ 前向传播
            logits, _, _ = self.model(x)
            
            print(f"[CAM 调试] Logit 值: {logits.item():.6f}")

            # ✅ 计算用于梯度的得分
            # 对于异常检测，我们想要最大化 logit（越接近 1 的 sigmoid 越好）
            score = logits.sum()
            
            print(f"[CAM 调试] Score: {score.item():.6f}")
            print(f"[CAM 调试] Score requires_grad: {score.requires_grad}")
            
            # ✅ 反向传播 - 关键修复：不要在这之前调用 zero_grad()
            # 这是之前代码的致命错误！
            score.backward(retain_graph=True)
            
            print(f"[CAM 调试] 梯度计算完成")

        # 获取梯度和特征
        grads = self.gradients  # [B, C, H, W]
        fmap = self.features    # [B, C, H, W]

        print(f"[CAM 调试] grads: {'存在' if grads is not None else 'None'} shape={grads.shape if grads is not None else 'N/A'}")
        print(f"[CAM 调试] fmap: {'存在' if fmap is not None else 'None'} shape={fmap.shape if fmap is not None else 'N/A'}")
        
        if grads is not None:
            print(f"[CAM 调试] 梯度范围: min={grads.min():.6f}, max={grads.max():.6f}, mean={grads.mean():.6f}")
        if fmap is not None:
            print(f"[CAM 调试] 特征范围: min={fmap.min():.6f}, max={fmap.max():.6f}, mean={fmap.mean():.6f}")

        if grads is None or fmap is None:
            print("[错误] 梯度或特征为None - 这表示 hook 没有正确捕获数据")
            print(f"  - grads 是否为 None: {grads is None}")
            print(f"  - fmap 是否为 None: {fmap is None}")
            cam = torch.zeros(x.size(0), 1, 256, 256, device=x.device)
            self.remove_hooks()
            return cam

        # 改进：标准 Grad-CAM++ 实现，保留空间维度信息
        # alpha_kc = grad^2 / (2*grad^2 + sum_spatial(f * grad^3))
        
        # 计算第二阶导数项
        grad_2 = grads ** 2
        grad_3 = grads ** 3
        
        # 分母：需要在空间维度求和，但要保留通道维度
        # sum_{i,j} (f_c^{ij} * grad_c^{ij,3})
        sum_spatial_term = (fmap * grad_3).sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # 计算权重系数 alpha_kc，保留在每个通道的空间位置
        alpha_kc = grad_2 / (2 * grad_2 + sum_spatial_term + 1e-8)  # [B, C, H, W]
        
        # 计算加权梯度：ReLU(1 + sum_spatial(grad))
        # 确保只对正梯度做权重
        relu_grads = F.relu(grads)  # [B, C, H, W]
        
        # 加权特征：alpha * ReLU(grad) * feature
        weighted_fmap = alpha_kc * relu_grads * fmap  # [B, C, H, W]
        
        # 对通道维度求和得到 CAM
        cam = weighted_fmap.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # ReLU激活（只保留正激活）
        cam = F.relu(cam)
        
        # 双线性插值到原始尺寸
        cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
        cam_min = cam.min()
        cam_max = cam.max()
        
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = torch.zeros_like(cam)

        # 清理钩子
        self.remove_hooks()

        return cam