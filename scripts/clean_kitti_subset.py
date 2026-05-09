"""Step 1: 扫描 KITTI 序列，按 frame_id 精确匹配图像和点云。"""

import os
import cv2
import numpy as np
import pandas as pd

kitti_root = r"E:\monodepth-obstacle-warning\data\kitti"
save_dir = r"E:\monodepth-obstacle-warning\data\kitti"
os.makedirs(save_dir, exist_ok=True)

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

print(f"Found {len(sequences)} sequences")

rows = []
total_skipped = 0
for date_dir, seq, img_dir, lidar_dir in sequences:
    img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    lidar_set = set(f for f in os.listdir(lidar_dir) if f.endswith(".bin"))

    skipped = 0
    for img_name in img_names:
        frame_id = os.path.splitext(img_name)[0]
        lidar_name = frame_id + ".bin"
        if lidar_name not in lidar_set:
            skipped += 1
            continue

        img_path = os.path.join(img_dir, img_name)
        lidar_path = os.path.join(lidar_dir, lidar_name)

        img = cv2.imread(img_path)
        img_ok = img is not None
        h, w = img.shape[:2] if img_ok else (-1, -1)

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

    if skipped > 0:
        total_skipped += skipped
        print(f"  [SKIP] {seq}: {skipped} images have no matching lidar")

df = pd.DataFrame(rows)
csv_path = os.path.join(save_dir, "kitti_subset_cleaned.csv")
json_path = os.path.join(save_dir, "kitti_subset_cleaned.json")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
df.to_json(json_path, orient="records", force_ascii=False, indent=2)

print(f"\nTotal samples: {len(df)}")
print(f"Images OK: {(df['image_ok']==True).sum()}")
print(f"Lidars OK:  {(df['lidar_ok']==True).sum()}")
print(f"Skipped (no lidar match): {total_skipped}")
print(f"Saved: {csv_path}")
print(f"Saved: {json_path}")
