import os
import cv2
import numpy as np
import pandas as pd

root = r"E:\monodepth_project\data\kitti\2011_09_26"
sequences = [
    "2011_09_26_drive_0001_sync",
    "2011_09_26_drive_0002_sync",
    "2011_09_26_drive_0005_sync"
]

rows = []

for seq in sequences:
    img_dir = os.path.join(root, seq, "image_02", "data")
    lidar_dir = os.path.join(root, seq, "velodyne_points", "data")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".bin")])

    n = min(len(img_files), len(lidar_files))

    for i in range(n):
        img_name = img_files[i]
        lidar_name = lidar_files[i]

        img_path = os.path.join(img_dir, img_name)
        lidar_path = os.path.join(lidar_dir, lidar_name)

        img = cv2.imread(img_path)
        img_ok = img is not None

        if img_ok:
            h, w = img.shape[:2]
        else:
            h, w = -1, -1

        try:
            lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            lidar_points = lidar.shape[0]
            lidar_ok = True
        except Exception:
            lidar_points = -1
            lidar_ok = False

        rows.append({
            "sequence": seq,
            "frame_id": os.path.splitext(img_name)[0],
            "image_path": img_path,
            "lidar_path": lidar_path,
            "image_ok": img_ok,
            "lidar_ok": lidar_ok,
            "image_height": h,
            "image_width": w,
            "lidar_points": lidar_points
        })

df = pd.DataFrame(rows)

save_dir = r"E:\monodepth_project\data\kitti"
os.makedirs(save_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "kitti_subset_cleaned.csv")
json_path = os.path.join(save_dir, "kitti_subset_cleaned.json")

df.to_csv(csv_path, index=False, encoding="utf-8-sig")
df.to_json(json_path, orient="records", force_ascii=False, indent=2)

print("已生成：")
print(csv_path)
print(json_path)
print("\n前5行：")
print(df.head())
print("\n总样本数：", len(df))