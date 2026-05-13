"""Step 5: 生成数据集分析图表。"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KITTI_ROOT, RESULTS_DIR
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 路径 ===
csv_path = os.path.join(KITTI_ROOT, "kitti_subset_final_cleaned.csv")
plot_dir = os.path.join(RESULTS_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

THRESHOLD = 100000

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

df = pd.read_csv(csv_path)
print(f"加载 {len(df)} 条样本")

# 图1: 每序列样本数
seq_col = "sequence"
if "date" in df.columns:
    df["label"] = df["date"].astype(str) + "/" + df["sequence"]
    seq_col = "label"

counts = df[seq_col].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 5))
counts.plot(kind="bar", ax=ax)
ax.set_title("Samples per Sequence")
ax.set_xlabel("Sequence")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45, labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "sequence_counts_bar.png"), dpi=150)
plt.close(fig)

# 图2: 点云点数分布直方图
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["lidar_points"], bins=30, edgecolor="black")
ax.axvline(THRESHOLD, color="red", linestyle="--",
           label=f"Threshold ({THRESHOLD})")
ax.set_title("Distribution of LiDAR Point Counts")
ax.set_xlabel("LiDAR Points")
ax.set_ylabel("Frequency")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "lidar_points_hist.png"), dpi=150)
plt.close(fig)

# 图3: 点云点数趋势线
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["lidar_points"].values, linewidth=0.5)
ax.axhline(THRESHOLD, color="red", linestyle="--", alpha=0.5)
ax.set_title("LiDAR Point Counts by Sample Index")
ax.set_xlabel("Sample Index")
ax.set_ylabel("LiDAR Points")
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "lidar_points_line.png"), dpi=150)
plt.close(fig)

# 图4: 数据集划分饼图
if "split" in df.columns:
    split_counts = df["split"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"train": "#4CAF50", "val": "#FF9800", "test": "#F44336"}
    wedge_colors = [colors.get(k, "#999") for k in split_counts.index]
    ax.pie(split_counts.values, labels=split_counts.index, autopct="%1.1f%%",
           colors=wedge_colors, startangle=90)
    ax.set_title("Train / Val / Test Split")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "split_pie.png"), dpi=150)
    plt.close(fig)

print(f"图表已保存到 {plot_dir}")
