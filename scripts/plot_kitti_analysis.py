"""Generate Chinese dataset analysis plots for the cleaned KITTI subset."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KITTI_ROOT, RESULTS_DIR


csv_path = os.path.join(KITTI_ROOT, "kitti_subset_final_cleaned.csv")
plot_dir = os.path.join(RESULTS_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

LIDAR_POINTS_THRESHOLD = 100000

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

df = pd.read_csv(csv_path)
print(f"已加载 {len(df)} 个样本")

seq_col = "sequence"
if "date" in df.columns:
    df["label"] = df["date"].astype(str) + "/" + df["sequence"]
    seq_col = "label"

counts = df[seq_col].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 5))
counts.plot(kind="bar", ax=ax)
ax.set_title("各驾驶序列样本数量")
ax.set_xlabel("驾驶序列")
ax.set_ylabel("样本数量")
ax.tick_params(axis="x", rotation=45, labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "sequence_counts_bar.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["lidar_points"], bins=30, edgecolor="black")
ax.axvline(
    LIDAR_POINTS_THRESHOLD,
    color="red",
    linestyle="--",
    label=f"清洗阈值（{LIDAR_POINTS_THRESHOLD}）",
)
ax.set_title("LiDAR 点云数量分布")
ax.set_xlabel("LiDAR 点数")
ax.set_ylabel("样本频数")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "lidar_points_hist.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["lidar_points"].values, linewidth=0.5)
ax.axhline(LIDAR_POINTS_THRESHOLD, color="red", linestyle="--", alpha=0.5)
ax.set_title("各样本 LiDAR 点云数量变化")
ax.set_xlabel("样本索引")
ax.set_ylabel("LiDAR 点数")
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "lidar_points_line.png"), dpi=150)
plt.close(fig)

if "split" in df.columns:
    split_counts = df["split"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"train": "#4CAF50", "val": "#FF9800", "test": "#F44336"}
    wedge_colors = [colors.get(k, "#999999") for k in split_counts.index]
    label_map = {"train": "训练集", "val": "验证集", "test": "测试集"}
    labels = [label_map.get(k, k) for k in split_counts.index]
    ax.pie(
        split_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=wedge_colors,
        startangle=90,
    )
    ax.set_title("训练集 / 验证集 / 测试集划分")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "split_pie.png"), dpi=150)
    plt.close(fig)

print(f"图表已保存到 {plot_dir}")
