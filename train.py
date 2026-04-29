import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from config import Config
from datasets.kolektor import KolektorDataset
from models.wsl_net import WSLNet
from utils.losses import *
import time
from datetime import datetime

def plot_loss_curves(history):
    """
    Plot and save loss curves
    - Total loss
    - Individual loss components (Weighted BCE, Focal, HS)
    - Learning rate changes
    """
    os.makedirs(Config.output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss and Learning Rate Curves', fontsize=16, fontweight='bold')
    
    # Subplot 1: Total loss
    ax1 = axes[0, 0]
    ax1.plot(history['epoch'], history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Subplot 2: Individual loss components
    ax2 = axes[0, 1]
    def safe_plot(x, y, color, label):
        if len(x) == len(y) and len(y) > 0:
            ax2.plot(x, y, color, linewidth=2, label=label)
        else:
            print(f"Warning: skipping {label} because x/y lengths differ ({len(x)} vs {len(y)})")

    safe_plot(history['epoch'], history.get('weighted_bce_loss', []), 'r-', 'Weighted BCE Loss')
    safe_plot(history['epoch'], history.get('focal_loss', []), 'g-', 'Focal Loss')
    safe_plot(history['epoch'], history.get('hs_loss', []), 'orange', 'HS Loss')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Loss Components', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Subplot 3: Learning rate schedule
    ax3 = axes[1, 0]
    ax3.plot(history['epoch'], history['lr'], 'purple', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Learning Rate', fontsize=11)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Subplot 4: Smoothed total loss
    ax4 = axes[1, 1]
    ax4.plot(history['epoch'], history['total_loss'], 'b-', linewidth=1, alpha=0.5, label='Raw Loss')
    
    if len(history['total_loss']) > 10:
        window = 10
        smooth_loss = [sum(history['total_loss'][max(0, i-window):i+1]) / len(history['total_loss'][max(0, i-window):i+1]) 
                       for i in range(len(history['total_loss']))]
        ax4.plot(history['epoch'], smooth_loss, 'b-', linewidth=2, label=f'Smoothed Loss (window={window})')
    
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_title('Smoothed Total Loss Trend', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    loss_curve_path = os.path.join(Config.output_dir, 'loss_curves.png')
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Loss curves saved to {loss_curve_path}")
    
    pdf_path = os.path.join(Config.output_dir, 'loss_curves.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Loss curves (PDF) saved to {pdf_path}")
    
    plt.close()


def train(dataset_root=None, dataset_type="auto"):
    device = Config.device

    if dataset_root is None:
        dataset_root = Config.data_root
    
    dataset = KolektorDataset(dataset_root, img_size=Config.img_size, train=True, dataset_type=dataset_type)
    # 优化：增加num_workers到4-8，增加batch_size，启用更好的prefetch
    loader = DataLoader(
        dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=8,  # 增加从2到8
        pin_memory=True, 
        prefetch_factor=4,  # 增加从2到4
        persistent_workers=True  # 保持worker进程活跃
    )

    model = WSLNet().to(device)

    # 使用AdamW优化器，weight_decay有助于正则化
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    
    # 学习率调度器：余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr * 0.01)

    bce = torch.nn.BCEWithLogitsLoss()
    focal = FocalLoss()

    # Initialize loss history
    history = {
        'epoch': [],
        'total_loss': [],
        'weighted_bce_loss': [],
        'focal_loss': [],
        'hs_loss': [],
        'lr': [],
        'time': []
    }

    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"{'='*70}")
    print(f"Total epochs: {Config.epochs}")
    print(f"Learning rate (initial): {Config.lr:.6f}")
    print(f"Batch size: {Config.batch_size}")
    # print(f"Positive weight (for anomalies): {Config.pos_weight}")
    # print(f"Loss weights: BCE={Config.loss_weight_bce}, Focal={Config.loss_weight_focal}, HS={Config.loss_weight_hs}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    os.makedirs(Config.output_dir, exist_ok=True)
    
    # 初始化日志文件
    log_file = os.path.join(Config.output_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Training Configuration:\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total epochs: {Config.epochs}\n")
        f.write(f"Learning rate (initial): {Config.lr:.6f}\n")
        f.write(f"Batch size: {Config.batch_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset root: {dataset_root}\n")
        f.write(f"Dataset type: {dataset_type}\n")
        f.write(f"{'='*70}\n\n")

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        total_bce_loss = 0
        total_focal_loss = 0
        total_hs_loss = 0
        epoch_start = time.time()

        for img, label in loader:
            img = img.to(device)
            label = label.unsqueeze(1).to(device)

            logits, feat_vec, _ = model(img)

            # 多任务损失
            loss_bce = Config.bce_weight * bce(logits, label)
            loss_focal = Config.focal_weight * focal(logits, label)
            loss_hs = Config.hs_weight * hypersphere_loss(feat_vec, label.squeeze(), model.center)

            loss = loss_bce + loss_focal + loss_hs

            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            total_bce_loss += loss_bce.item()
            total_focal_loss += loss_focal.item()
            total_hs_loss += loss_hs.item()

        avg_loss = total_loss / len(loader)
        avg_bce = total_bce_loss / len(loader)
        avg_focal = total_focal_loss / len(loader)
        avg_hs = total_hs_loss / len(loader)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        log_msg = f"Epoch {epoch+1:3d}/{Config.epochs}: Loss={avg_loss:.4f} (BCE={avg_bce:.4f}, Focal={avg_focal:.4f}, HS={avg_hs:.4f}), LR={current_lr:.6f}, Time={epoch_time:.1f}s"
        print(log_msg)
        
        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")

        history['epoch'].append(epoch + 1)
        history['total_loss'].append(avg_loss)
        history['weighted_bce_loss'].append(avg_bce)
        history['focal_loss'].append(avg_focal)
        history['hs_loss'].append(avg_hs)
        history['lr'].append(current_lr)
        history['time'].append(epoch_time)
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(Config.output_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_msg = f"  ✓ Checkpoint saved: {checkpoint_path}"
            print(checkpoint_msg)
            with open(log_file, 'a') as f:
                f.write(checkpoint_msg + "\n")

        scheduler.step()

    plot_loss_curves(history)

    # torch.save(model.state_dict(), "model.pth")
    torch.save(model.state_dict(), Config.model_path)
    total_time = sum(history['time'])
    
    final_msg = f"\n✓ Model saved to {Config.model_path}\n✓ Total training time: {total_time:.1f}s ({total_time/60:.1f}m)"
    print(final_msg)
    
    # 写入最终总结到日志
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Training Complete!\n")
        f.write(f"{'='*70}\n")
        f.write(final_msg + "\n")
        f.write(f"Training End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log file: {log_file}\n")
    
    


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train WSL model on defect detection dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset root path (default: Config.data_root)")
    parser.add_argument("--dataset-type", type=str, default="auto", choices=["auto", "kolektor_sdd", "kolektor_sdd2"],
                        help="Dataset type (default: auto-detect)")
    args = parser.parse_args()
    
    train(dataset_root=args.dataset, dataset_type=args.dataset_type)