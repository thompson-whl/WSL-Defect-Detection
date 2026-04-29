import torch


class Config:
    data_roots = ["./KolektorSDD", "./KolektorSDD2", "./mvtecad"]  # 支持多个数据集路径
    data_root = data_roots[1]  # 默认使用第一个数据集路径

    # 输出目录
    output_dir = "./outputs"
    cam_dir = "./outputs/cam"
    mask_dir = "./outputs/mask"
    model_path = "./outputs/model.pth"

    img_size = 256
    batch_size = 8
    epochs = 1
    lr = 1e-3
    weight_decay = 1e-4

    # 损失函数权重
    bce_weight=1.0
    focal_weight=1.0
    hs_weight=0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cam_threshold = "otsu"