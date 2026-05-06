"""Step 1: 提取 KITTI 序列元数据，检查图像和点云可读性。"""

import os
import cv2
import numpy as np
import pandas as pd

# === 路径设置 ===
kitti_root = r"E:\monodepth_project\data\kitti"
save_dir = r"E:\monodepth-obstacle-warning\data\kitti"
os.makedirs(save_dir, exist_ok=True)

# === 自动发现所有序列 ===
sequences = []
for date_dir in sorted(os.listdir(kitti_root)):
    date_path = os.path.join(kitti_root, date_dir)
    if not os.path.isdir(date_path):
        continue
    for seq in sorted(os.listdir(date_path)):
        if seq.endswith("_sync"):
            img_dir = os.path.join(date_path, seq, "image_02", "data")
            lidar_dir = os.path.join(date_path, seq, "velodyne_points", "data")
            if os.path.isdir(img_dir) and os.path.isdir(lidar_dir):
                sequences.append((date_dir, seq, img_dir, lidar_dir))

print(f"发现 {len(sequences)} 个序列")

rows = []
for date_dir, seq, img_dir, lidar_dir in sequences:
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".bin")])

    n = min(len(img_files), len(lidar_files))
    if len(img_files) != len(lidar_files):
        print(f"  [注意] {seq}: 图像 {len(img_files)} vs 点云 {len(lidar_files)}, 取 {n} 对")

    for i in range(n):
        img_name = img_files[i]
        lidar_name = lidar_files[i]
        frame_id = os.path.splitext(img_name)[0]

        img_path = os.path.join(img_dir, img_name)
        lidar_path = os.path.join(lidar_dir, lidar_name)

        # 检查图像
        img = cv2.imread(img_path)
        img_ok = img is not None
        h, w = img.shape[:2] if img_ok else (-1, -1)

        # 检查点云
        try:
            lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            lidar_points = lidar.shape[0]
            lidar_ok = True
        except Exception:
            lidar_points = -1
            lidar_ok = False

        rows.append({
            "date": date_dir,
            "sequence": seq,
            "frame_id": frame_id,
            "image_path": img_path,
            "lidar_path": lidar_path,
            "image_ok": img_ok,
            "lidar_ok": lidar_ok,
            "image_height": h,
            "image_width": w,
            "lidar_points": lidar_points,
        })

df = pd.DataFrame(rows)

csv_path = os.path.join(save_dir, "kitti_subset_cleaned.csv")
json_path = os.path.join(save_dir, "kitti_subset_cleaned.json")

df.to_csv(csv_path, index=False, encoding="utf-8-sig")
df.to_json(json_path, orient="records", force_ascii=False, indent=2)

print(f"\n总样本数: {len(df)}")
print(f"图像可读: {(df['image_ok']==True).sum()}")
print(f"点云可读: {(df['lidar_ok']==True).sum()}")
print(f"已保存: {csv_path}")
print(f"已保存: {json_path}")
