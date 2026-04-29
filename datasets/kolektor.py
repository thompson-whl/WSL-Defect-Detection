import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler


class KolektorDataset(Dataset):
    def __init__(self, root, img_size=256, train=True, dataset_type="auto"):
        """
        KolektorSDD/KolektorSDD2 dataset loader
        
        KolektorSDD Structure:
        - 50 directories: kos01, kos02, ..., kos50
        - Each directory: 8 image-label pairs (PartX.jpg + PartX_label.bmp)
        - Labels: BMP file with pixel-level annotations
        
        KolektorSDD2 Structure:
        - train/ and test/ subdirectories
        - Images: XXXXX.png with labels XXXXX_GT.png
        - Labels: PNG file with pixel-level annotations
          - Non-zero pixels = defect area (anomaly)
          - All-zero PNG = normal image
        
        Args:
            root: Path to dataset root directory
            img_size: Size to resize images to
            train: Whether to use training or test set (used for KolektorSDD2)
            dataset_type: "kolektor_sdd" or "kolektor_sdd2" or "auto" (auto-detect)
        """
        self.imgs = []
        self.labels = []
        self.label_maps = []
        self.img_size = img_size
        self.dataset_type = dataset_type
        self._label_cache = {}  # 缓存标签以避免重复读取
        
        # Auto-detect dataset type
        if dataset_type == "auto":
            if os.path.isdir(os.path.join(root, "train")):
                self.dataset_type = "kolektor_sdd2"
            elif os.path.isdir(os.path.join(root, "kos01")):
                self.dataset_type = "kolektor_sdd"
            else:
                raise ValueError(f"Unknown dataset structure in {root}")
        
        print(f"Loading dataset type: {self.dataset_type}")
        
        if self.dataset_type == "kolektor_sdd":
            self._load_kolektor_sdd(root)
        elif self.dataset_type == "kolektor_sdd2":
            self._load_kolektor_sdd2(root, train)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        print(f"\nDataset Summary:")
        print(f"  Total images: {len(self.imgs)}")
        print(f"  Normal samples: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Anomaly samples: {sum(1 for l in self.labels if l == 1)}")
        
        # Compute sample weights for WeightedRandomSampler
        self.sample_weights = self._compute_sample_weights()
    
    def _load_kolektor_sdd(self, root):
        """Load KolektorSDD dataset"""
        # Get all directories (kos01, kos02, ..., kos50)
        folders = sorted([f for f in os.listdir(root) if f.startswith('kos')])
        
        print(f"Found {len(folders)} directories")
        
        for folder in folders:
            folder_path = os.path.join(root, folder)
            
            if not os.path.isdir(folder_path):
                continue
            
            # Find all jpg files in this directory
            jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            
            for jpg_file in jpg_files:
                img_path = os.path.join(folder_path, jpg_file)
                
                # Construct corresponding label path
                # PartX.jpg -> PartX_label.bmp
                base_name = os.path.splitext(jpg_file)[0]
                label_path = os.path.join(folder_path, f"{base_name}_label.bmp")
                
                # Check if label file exists
                if not os.path.exists(label_path):
                    print(f"Warning: Label not found for {img_path}, skipping...")
                    continue
                
                # Determine if this is an anomaly by checking if bmp has non-zero pixels
                label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                
                if label_img is None:
                    print(f"Warning: Could not read label {label_path}, skipping...")
                    continue
                
                has_defect = (label_img > 0).any()
                label = 1 if has_defect else 0
                
                self.imgs.append(img_path)
                self.labels.append(label)
                self.label_maps.append(label_path)
    
    def _load_kolektor_sdd2(self, root, train=True):
        """Load KolektorSDD2 dataset"""
        # Choose train or test set
        split_dir = "train" if train else "test"
        split_path = os.path.join(root, split_dir)
        
        if not os.path.isdir(split_path):
            raise ValueError(f"KolektorSDD2 {split_dir} directory not found at {split_path}")
        
        # Find all PNG files (images)
        png_files = sorted([f for f in os.listdir(split_path) if f.endswith('.png') and not f.endswith('_GT.png')])
        
        print(f"Found {len(png_files)} images in {split_dir} set")
        
        for png_file in png_files:
            img_path = os.path.join(split_path, png_file)
            
            # Construct corresponding label path
            # XXXXX.png -> XXXXX_GT.png
            base_name = os.path.splitext(png_file)[0]
            label_path = os.path.join(split_path, f"{base_name}_GT.png")
            
            # Check if label file exists
            if not os.path.exists(label_path):
                print(f"Warning: Label not found for {img_path}, skipping...")
                continue
            
            # Fast label detection: read only to check for defects
            # 优化：避免重复读取标签，只在__getitem__时读取
            self.imgs.append(img_path)
            self.label_maps.append(label_path)
            
            # 延迟读取标签，在第一次访问时缓存
            # For now, mark as to-be-determined
            self.labels.append(-1)  # -1 表示未读取
        
        # 在加载后扫描一遍确定标签（快速扫描）
        self._scan_labels_fast()

    def _scan_labels_fast(self):
        """快速扫描标签，确定normal/anomaly分类"""
        print("Scanning labels quickly...")
        for i, label_path in enumerate(self.label_maps):
            if self.labels[i] != -1:  # Already determined (KolektorSDD)
                continue
            
            label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label_img is not None:
                has_defect = (label_img > 0).any()
                self.labels[i] = 1 if has_defect else 0
            else:
                self.labels[i] = 0  # Default to normal if can't read

    def _compute_sample_weights(self):
        """
        Compute sample weights for WeightedRandomSampler
        - Anomaly samples (label=1) get higher weight
        - Normal samples (label=0) get lower weight
        This ensures anomaly samples appear more frequently during sampling
        """
        num_samples = len(self.labels)
        num_positives = sum(1 for l in self.labels if l == 1)  # anomaly count
        num_negatives = num_samples - num_positives  # normal count
        
        weights = []
        for label in self.labels:
            if label == 1:  # anomaly
                # Anomaly weight: total_samples / (2 * anomaly_count)
                # This increases the expected number of times anomaly samples are sampled per epoch
                weight = num_samples / (2 * max(num_positives, 1))
            else:  # normal
                # Normal weight: total_samples / (2 * normal_count)
                weight = num_samples / (2 * max(num_negatives, 1))
            weights.append(weight)
        
        print(f"\nSample weight statistics:")
        if any(l == 0 for l in self.labels):
            normal_idx = next(i for i, l in enumerate(self.labels) if l == 0)
            print(f"  - Normal sample weight: {weights[normal_idx]:.4f}")
        if any(l == 1 for l in self.labels):
            anomaly_idx = next(i for i, l in enumerate(self.labels) if l == 1)
            print(f"  - Anomaly sample weight: {weights[anomaly_idx]:.4f}")
        
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Get image and label (optimized for speed)
        
        Returns:
            img: [3, H, W] normalized image tensor
            label: scalar tensor (0 or 1)
        """
        img = cv2.imread(self.imgs[idx])
        
        if img is None:
            raise RuntimeError(f"Could not read image: {self.imgs[idx]}")
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img[:, :, ::-1] / 255.0  # BGR to RGB, normalize to [0, 1]
        
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        # 优化：使用from_numpy而不是tensor，速度快
        img = torch.from_numpy(img).float()
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return img, label


# def create_weighted_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
#     """
#     Create DataLoader with WeightedRandomSampler
#     This oversamples anomaly samples during training to handle class imbalance
    
#     Args:
#         dataset: KolektorDataset instance
#         batch_size: Batch size
#         shuffle: Not used (sampler handles sampling)
#         num_workers: Number of worker processes for data loading
    
#     Returns:
#         DataLoader with weighted sampling
#     """
#     sampler = WeightedRandomSampler(
#         weights=dataset.sample_weights,
#         num_samples=len(dataset),
#         replacement=True  # allow repeated sampling
#     )
    
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,  # use weighted sampler instead of shuffle
#         num_workers=num_workers,  # 优化: 多线程数据加载
#         pin_memory=True,  # 优化: 锁定内存以加快GPU传输
#         prefetch_factor=2  # 优化: 预加载下一批数据
#     )
    
#     return loader
