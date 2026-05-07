"""Convert our split files to Monodepth2 format.

Our format:  image_path lidar_path
Monodepth2:  date/seq frame_id side

Output split files go directly to Monodepth2's splits/kitti_subset/
"""

import pandas as pd
import os

csv_path = r"E:\monodepth-obstacle-warning\data\kitti\kitti_subset_final_cleaned.csv"
md2_split_dir = r"E:\monodepth-obstacle-warning\monodepth2\splits\kitti_subset"

df = pd.read_csv(csv_path)
print(f"Loading {len(df)} entries")

for split_name in ["train", "val", "test"]:
    subset = df[df["split"] == split_name].copy()
    out_path = os.path.join(md2_split_dir, f"{split_name}_files.txt")

    # Remove first and last frame of each sequence (no context frames available)
    keep_mask = pd.Series(True, index=subset.index)
    for seq, grp in subset.groupby("sequence"):
        min_fid = grp["frame_id"].min()
        max_fid = grp["frame_id"].max()
        drop_idx = grp[(grp["frame_id"] == min_fid) | (grp["frame_id"] == max_fid)].index
        keep_mask[drop_idx] = False
    subset = subset[keep_mask]

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in subset.iterrows():
            # Monodepth2 format: date/drive frame_id side
            entry = f"{row['date']}/{row['sequence']} {int(row['frame_id'])} l"
            f.write(entry + "\n")

    print(f"{split_name}: {len(subset)} entries -> {out_path}")
