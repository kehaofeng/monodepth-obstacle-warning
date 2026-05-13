"""工具: 从清洗后的 CSV 生成 train/val/test 文件列表。"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KITTI_ROOT
import pandas as pd

csv_path = os.path.join(KITTI_ROOT, "kitti_subset_final_cleaned.csv")
save_dir = KITTI_ROOT

df = pd.read_csv(csv_path)

if "split" not in df.columns:
    print("错误: CSV 中没有 split 列，请先运行 split_dataset.py")
    exit()

for split_name in ["train", "val", "test"]:
    subset = df[df["split"] == split_name]
    filepath = os.path.join(save_dir, f"{split_name}_files.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        for _, row in subset.iterrows():
            f.write(f"{row['image_path']} {row['lidar_path']}\n")
    print(f"{split_name}: {len(subset)} 条 -> {filepath}")
