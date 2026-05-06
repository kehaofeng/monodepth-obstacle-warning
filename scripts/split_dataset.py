"""Step 3: 划分训练集/验证集/测试集，生成训练文件列表。

策略：序列从大到小排列，按 train/val/test/train/val/test... 轮流分配，
确保每个集合都分到数据。
"""

import os
import pandas as pd

# === 路径设置 ===
final_csv = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_final_cleaned.csv"
save_dir = r"E:\monodepth-obstacle-warning\data\kitti"

df = pd.read_csv(final_csv)
print(f"加载 {len(df)} 条清洗后样本")

# 按序列分组，样本多的排前面
seq_sizes = df.groupby("sequence").size()
seq_names = sorted(seq_sizes.keys(), key=lambda s: seq_sizes[s], reverse=True)

# 轮流分配: train, val, test, train, val, test, ...
pools = ["train", "val", "test"]
assignment = {}
counts = {"train": 0, "val": 0, "test": 0}

for i, seq in enumerate(seq_names):
    pool = pools[i % 3]
    assignment[seq] = pool
    counts[pool] += seq_sizes[seq]

df["split"] = df["sequence"].map(assignment)
total = len(df)

# === 输出 ===
print(f"\n=== 划分结果 ===")
for split_name in ["train", "val", "test"]:
    subset = df[df["split"] == split_name]
    print(f"{split_name}: {len(subset)} 条 ({len(subset)/total*100:.1f}%)")
    for seq in sorted(subset["sequence"].unique()):
        print(f"  {seq}: {len(subset[subset['sequence']==seq])}")

for split_name in ["train", "val", "test"]:
    subset = df[df["split"] == split_name]
    filepath = os.path.join(save_dir, f"{split_name}_files.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        for _, row in subset.iterrows():
            f.write(f"{row['image_path']} {row['lidar_path']}\n")
    print(f"  -> {filepath} ({len(subset)} 行)")

df.to_csv(final_csv, index=False, encoding="utf-8-sig")
print(f"\n已更新 {final_csv}")
