import torch


class Config:
    data_roots = ["./KolektorSDD", "./KolektorSDD2", "./mvtecad"]  # 支持多个数据集路径
    data_root = data_roots[2]  

    # MVTecAD配置
    mvtecad_category_list = ["bottle", "cable", "capsule", "carpet", "grid", 
    "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", 
    "transistor", "wood", "zipper"]
    mvtecad_category = mvtecad_category_list[0]  # 可选: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

    # 输出目录
    output_dir = "./outputs"
    cam_dir = "./outputs/cam"
    mask_dir = "./outputs/mask"
    model_path = "./outputs/model.pth"


    # KolektorSDD/KolektorSDD2：256x512，MVTecAD：256x256
    img_width = 256
    img_height = 512
    batch_size = 32
    epochs = 50
    lr = 1e-3
    weight_decay = 1e-4

    # 损失函数权重
    focal_weight=2.0
    hs_weight=0.01

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cam_threshold = "otsu"
