import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"E:\monodepth_project\data\kitti\kitti_subset_final_cleaned.csv"
save_dir = r"E:\monodepth_project\results\plots"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# 图1：每个序列样本数量柱状图
seq_counts = df["sequence"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
seq_counts.plot(kind="bar")
plt.title("Number of Samples per Sequence")
plt.xlabel("Sequence")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "sequence_counts_bar.png"))
plt.close()

# 图2：点云点数分布直方图
plt.figure(figsize=(8, 5))
plt.hist(df["lidar_points"], bins=20)
plt.title("Distribution of LiDAR Point Counts")
plt.xlabel("LiDAR Points")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lidar_points_hist.png"))
plt.close()

# 图3：点云点数随样本序号变化折线图
plt.figure(figsize=(10, 5))
plt.plot(df["lidar_points"].values)
plt.title("LiDAR Point Counts by Sample Index")
plt.xlabel("Sample Index")
plt.ylabel("LiDAR Points")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lidar_points_line.png"))
plt.close()

print("图表已保存到：", save_dir)