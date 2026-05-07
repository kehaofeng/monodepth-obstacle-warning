"""工具: 可视化样本图像及其对应的深度图。"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 路径 ===
kitti_root = r"E:\monodepth-obstacle-warning\data\kitti"
csv_path = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_final_cleaned.csv"
plot_path = r"E:\monodepth-obstacle-warning\results\plots\sample_depth_grid.png"

# === 加载标定 ===
def read_calib(calib_path):
    calib = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            try:
                calib[key.strip()] = np.array([float(x) for x in val.strip().split()])
            except ValueError:
                pass
    return calib

# 只用 2011_09_26 的标定（所有序列共用）
calib_dir = os.path.join(kitti_root, "2011_09_26")
velo_calib = read_calib(os.path.join(calib_dir, "calib_velo_to_cam.txt"))
cam_calib = read_calib(os.path.join(calib_dir, "calib_cam_to_cam.txt"))

R = velo_calib["R"].reshape(3, 3)
t = velo_calib["T"].reshape(3, 1)
T_velo_cam = np.eye(4)
T_velo_cam[:3, :3] = R
T_velo_cam[:3, 3:4] = t

P = cam_calib["P_rect_02"].reshape(3, 4)
K = P[:, :3]

# === 取 4 个训练集样本 ===
df = pd.read_csv(csv_path)
train_df = df[df["split"] == "train"]
indices = np.linspace(0, len(train_df) - 1, 4, dtype=int)
samples = train_df.iloc[indices]

fig, axes = plt.subplots(4, 2, figsize=(12, 16))

for j, (_, row) in enumerate(samples.iterrows()):
    # 原图
    img = cv2.imread(row["image_path"])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 深度图 (简化投影)
    points = np.fromfile(row["lidar_path"], dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyz_cam = (T_velo_cam @ xyz_h.T).T[:, :3]
    front = xyz_cam[:, 2] > 0.1
    xyz_cam = xyz_cam[front]
    pts = (K @ xyz_cam.T).T
    u = np.round(pts[:, 0] / pts[:, 2]).astype(int)
    v = np.round(pts[:, 1] / pts[:, 2]).astype(int)
    z = pts[:, 2]
    valid = (u >= 0) & (u < 1242) & (v >= 0) & (v < 375)
    u, v, z = u[valid], v[valid], z[valid]
    depth = np.zeros((375, 1242), dtype=np.float32)
    if len(z) > 0:
        order = np.argsort(-z)
        depth[v[order], u[order]] = z[order]

    depth_vis = np.clip(depth, 0, 80) / 80.0

    axes[j, 0].imshow(img_rgb)
    axes[j, 0].set_title(f"{row['sequence']} frame {row['frame_id']}")
    axes[j, 0].axis("off")

    axes[j, 1].imshow(depth_vis, cmap="inferno")
    axes[j, 1].set_title(f"Depth (0~80m, {row['lidar_points']} LiDAR pts)")
    axes[j, 1].axis("off")

fig.tight_layout()
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"已保存: {plot_path}")
