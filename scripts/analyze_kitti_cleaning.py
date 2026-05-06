"""Step 2: 分析数据质量，检测异常样本，输出最终清洗数据集。"""

import os
import pandas as pd

# === 路径设置 ===
csv_path = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_cleaned.csv"
abnormal_path = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_abnormal.csv"
final_path = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_final_cleaned.csv"

# === 参数 ===
LIDAR_THRESHOLD = 100000   # 点云点数低于此值视为异常
EXPECTED_H, EXPECTED_W = 375, 1242  # KITTI 标准图像尺寸

# === 加载数据 ===
df = pd.read_csv(csv_path)
n_total = len(df)
print(f"加载样本: {n_total}")

# === 每序列样本数 ===
print("\n=== 各序列样本数 ===")
seq_counts = df.groupby(["date", "sequence"]).size()
print(seq_counts.to_string())

# === 质量检查 ===
n_bad_img = (~df["image_ok"]).sum()
n_bad_lidar = (~df["lidar_ok"]).sum()
n_wrong_size = ((df["image_height"] != EXPECTED_H) | (df["image_width"] != EXPECTED_W)).sum()
n_low_points = (df["lidar_points"] < LIDAR_THRESHOLD).sum()

print(f"\n=== 质量检查 ===")
print(f"图像读取失败:     {n_bad_img}")
print(f"点云读取失败:     {n_bad_lidar}")
print(f"尺寸异常 (应为{EXPECTED_H}x{EXPECTED_W}): {n_wrong_size}")
print(f"点云点数过低 (<{LIDAR_THRESHOLD}): {n_low_points}")

# === 点云统计 ===
print(f"\n=== 点云点数分布 ===")
print(df["lidar_points"].describe())

# === 标记异常样本 ===
abnormal_mask = (
    (~df["image_ok"]) |
    (~df["lidar_ok"]) |
    (df["image_height"] != EXPECTED_H) |
    (df["image_width"] != EXPECTED_W) |
    (df["lidar_points"] < LIDAR_THRESHOLD)
)
abnormal_df = df[abnormal_mask].copy()

print(f"\n=== 异常样本: {len(abnormal_df)} 条 ===")
for _, row in abnormal_df.iterrows():
    reasons = []
    if not row["image_ok"]: reasons.append("图像读取失败")
    if not row["lidar_ok"]: reasons.append("点云读取失败")
    if row["image_height"] != EXPECTED_H or row["image_width"] != EXPECTED_W:
        reasons.append(f"尺寸异常({row['image_height']}x{row['image_width']})")
    if row["lidar_points"] < LIDAR_THRESHOLD:
        reasons.append(f"点云过少({row['lidar_points']})")
    print(f"  {row['date']}/{row['sequence']} frame={row['frame_id']}: {', '.join(reasons)}")

abnormal_df.to_csv(abnormal_path, index=False, encoding="utf-8-sig")

# === 输出清洗后数据 ===
clean_df = df[~abnormal_mask].copy()
print(f"\n=== 清洗结果: {len(clean_df)} 条 (移除 {len(abnormal_df)} 条) ===")
clean_df.to_csv(final_path, index=False, encoding="utf-8-sig")
print(f"已保存: {final_path}")
