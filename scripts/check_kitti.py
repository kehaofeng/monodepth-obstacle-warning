import os
import cv2
import random
import numpy as np

root = r"E:\monodepth_project\data\kitti\2011_09_26"

sequences = [
    "2011_09_26_drive_0001_sync",
    "2011_09_26_drive_0002_sync",
    "2011_09_26_drive_0005_sync"
]

for seq in sequences:
    img_dir = os.path.join(root, seq, "image_02", "data")
    lidar_dir = os.path.join(root, seq, "velodyne_points", "data")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".bin")])

    print(f"\n序列: {seq}")
    print(f"图像数量: {len(img_files)}")
    print(f"点云数量: {len(lidar_files)}")

    sample_ids = random.sample(range(len(img_files)), min(3, len(img_files)))
    for idx in sample_ids:
        img_path = os.path.join(img_dir, img_files[idx])
        lidar_path = os.path.join(lidar_dir, lidar_files[idx])

        img = cv2.imread(img_path)
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        print(f"  样本 {idx}:")
        print(f"    图像: {img_files[idx]}, shape={img.shape if img is not None else None}")
        print(f"    点云: {lidar_files[idx]}, shape={lidar.shape}")