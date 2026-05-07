"""工具: 快速检查 KITTI 数据完整性（图像与点云是否正确配对）。"""

import os
import cv2
import random
import numpy as np

kitti_root = r"E:\monodepth-obstacle-warning\data\kitti"

# 自动发现序列
for date_dir in sorted(os.listdir(kitti_root)):
    date_path = os.path.join(kitti_root, date_dir)
    if not os.path.isdir(date_path):
        continue
    for seq in sorted(os.listdir(date_path)):
        if not seq.endswith("_sync"):
            continue
        img_dir = os.path.join(date_path, seq, "image_02", "data")
        lidar_dir = os.path.join(date_path, seq, "velodyne_points", "data")

        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".bin")])

        print(f"\n{seq}")
        print(f"  图像: {len(img_files)}  点云: {len(lidar_files)}")

        n = min(len(img_files), 3)
        for idx in sorted(random.sample(range(len(img_files)), n)):
            img = cv2.imread(os.path.join(img_dir, img_files[idx]))
            lidar = np.fromfile(os.path.join(lidar_dir, lidar_files[idx]), dtype=np.float32).reshape(-1, 4)
            status = "OK" if img is not None and lidar.shape[1] == 4 else "FAIL"
            print(f"  [{status}] {img_files[idx]}: img={img.shape if img is not None else None}, lidar={lidar.shape}")
