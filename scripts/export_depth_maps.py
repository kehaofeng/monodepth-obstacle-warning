"""Step 4: 利用标定参数将 LiDAR 点云投影为深度图（训练 ground truth）。

原理：
1. 读取 LiDAR 点云 (N x 4: x, y, z, 反射率)
2. 用标定矩阵转换到相机坐标系
3. 用相机内参投影到图像平面
4. 取每个像素最近点的深度值
"""

import os
import cv2
import numpy as np
import pandas as pd

# === 路径设置 ===
kitti_root = r"E:\monodepth-obstacle-warning\data\kitti"
final_csv = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_final_cleaned.csv"
depth_dir = r"E:\monodepth-obstacle-warning\data\kitti\depth_maps"  # 深度图太大，存在外部
os.makedirs(depth_dir, exist_ok=True)


def read_calib(calib_path):
    """读取 KITTI 标定文件，跳过时间戳等非数值行。"""
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
                pass  # 跳过非数值行（如时间戳）
    return calib


def get_velo_to_cam(calib_dir):
    """从标定文件读取 Velodyne -> Camera 的 4x4 变换矩阵。"""
    velo_path = os.path.join(calib_dir, "calib_velo_to_cam.txt")
    calib = read_calib(velo_path)
    R = calib["R"].reshape(3, 3)
    t = calib["T"].reshape(3, 1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


def get_camera_intrinsics(calib_dir, cam_id="02"):
    """从标定文件读取相机内参 K (3x3)。"""
    cam_path = os.path.join(calib_dir, "calib_cam_to_cam.txt")
    calib = read_calib(cam_path)
    P = calib[f"P_rect_{cam_id}"].reshape(3, 4)
    return P[:, :3]


def project_to_depth(lidar_path, T_velo_cam, K, img_h, img_w):
    """将 LiDAR 点云投影到图像平面，生成深度图。"""
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    # 变换到相机坐标系
    xyz = points[:, :3]
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    xyz_cam = (T_velo_cam @ xyz_h.T).T[:, :3]

    # 只保留相机前方的点 (z > 0.1m)
    front = xyz_cam[:, 2] > 0.1
    xyz_cam = xyz_cam[front]

    if len(xyz_cam) == 0:
        return np.zeros((img_h, img_w), dtype=np.float32)

    # 投影到图像平面: (u, v) = K * (x, y, z) / z
    pts = (K @ xyz_cam.T).T
    u = np.round(pts[:, 0] / pts[:, 2]).astype(int)
    v = np.round(pts[:, 1] / pts[:, 2]).astype(int)
    z = pts[:, 2]

    # 过滤图像边界外的点
    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, z = u[valid], v[valid], z[valid]

    # 生成深度图：同一像素保留最近的点（z 最小）
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    if len(z) > 0:
        order = np.argsort(-z)  # 从远到近排列，近的覆盖远的
        depth_map[v[order], u[order]] = z[order]

    return depth_map


# === 主流程 ===
df = pd.read_csv(final_csv)
print(f"加载 {len(df)} 条样本")

# 按日期加载标定参数
calib_cache = {}
for date in df["date"].unique():
    calib_dir = os.path.join(kitti_root, str(date))
    try:
        T = get_velo_to_cam(calib_dir)
        K = get_camera_intrinsics(calib_dir, "02")
        calib_cache[date] = (T, K)
        print(f"标定加载成功: {date}")
    except Exception as e:
        print(f"  标定加载失败 {date}: {e}")

# 生成深度图
depth_paths = []
for _, row in df.iterrows():
    date = str(row["date"])
    if date not in calib_cache:
        depth_paths.append("")
        continue

    T, K = calib_cache[date]
    depth_filename = f"{date}_{row['sequence']}_{row['frame_id']}.npy"
    depth_path = os.path.join(depth_dir, depth_filename)

    if not os.path.exists(depth_path):
        try:
            depth = project_to_depth(
                row["lidar_path"], T, K,
                int(row["image_height"]), int(row["image_width"])
            )
            np.save(depth_path, depth)
        except Exception as e:
            print(f"  失败 {row['sequence']}/{row['frame_id']}: {e}")
            depth_path = ""

    depth_paths.append(depth_path)

df["depth_path"] = depth_paths
n_ok = sum(1 for p in depth_paths if p)
print(f"\n深度图生成: {n_ok}/{len(df)}")
print(f"保存在: {depth_dir}")

df.to_csv(final_csv, index=False, encoding="utf-8-sig")
