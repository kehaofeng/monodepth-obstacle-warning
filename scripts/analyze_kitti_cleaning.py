"""Step 2: 分析数据质量，检测异常样本，输出最终清洗数据集。"""

import os
import pandas as pd

# === 路径设置 ===
csv_path = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_cleaned.csv"
abnormal_path = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_abnormal.csv"
final_path = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_final_cleaned.csv"

# === 参数 ===
LIDAR_THRESHOLD = 100000   # 点云点数低于此值视为异常

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
n_wrong_size = ((df["image_height"] <= 0) | (df["image_width"] <= 0)).sum()
n_low_points = (df["lidar_points"] < LIDAR_THRESHOLD).sum()

print(f"\n=== 质量检查 ===")
print(f"图像读取失败:     {n_bad_img}")
print(f"点云读取失败:     {n_bad_lidar}")
print(f"尺寸无效 (≤0):    {n_wrong_size}")
print(f"点云点数过低 (<{LIDAR_THRESHOLD}): {n_low_points}")

# 显示实际的尺寸分布
print(f"\n=== 图像尺寸分布 ===")
size_counts = df[["image_height", "image_width"]].value_counts()
for (h, w), c in size_counts.items():
    print(f"  {h}x{w}: {c} 条")

# === 点云统计 ===
print(f"\n=== 点云点数分布 ===")
print(df["lidar_points"].describe())

# === 标记异常样本 ===
abnormal_mask = (
    (~df["image_ok"]) |
    (~df["lidar_ok"]) |
    (df["image_height"] <= 0) |
    (df["image_width"] <= 0) |
    (df["lidar_points"] < LIDAR_THRESHOLD)
)
abnormal_df = df[abnormal_mask].copy()

print(f"\n=== 异常样本: {len(abnormal_df)} 条 ===")
for _, row in abnormal_df.iterrows():
    reasons = []
    if not row["image_ok"]: reasons.append("图像读取失败")
    if not row["lidar_ok"]: reasons.append("点云读取失败")
    if row["image_height"] <= 0 or row["image_width"] <= 0:
        reasons.append(f"尺寸无效({row['image_height']}x{row['image_width']})")
    if row["lidar_points"] < LIDAR_THRESHOLD:
        reasons.append(f"点云过少({row['lidar_points']})")
    print(f"  {row['date']}/{row['sequence']} frame={row['frame_id']}: {', '.join(reasons)}")

abnormal_df.to_csv(abnormal_path, index=False, encoding="utf-8-sig")

# === 输出清洗后数据 ===
clean_df = df[~abnormal_mask].copy()
print(f"\n=== 清洗结果: {len(clean_df)} 条 (移除 {len(abnormal_df)} 条) ===")
clean_df.to_csv(final_path, index=False, encoding="utf-8-sig")
print(f"已保存: {final_path}")
