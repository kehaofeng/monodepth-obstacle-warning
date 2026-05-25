"""Convert kitti_subset split files from full paths to Lite-Mono format.

Our format (from Monodepth2):
    E:\...\data\kitti\2011_09_26\2011_09_26_drive_0002_sync\image_02\data\0000000000.png <lidar_path>

Lite-Mono format:
    <seq>/<drive_sync> <frame_index> <side>
    e.g. 2011_09_26/2011_09_26_drive_0002_sync 0 l
"""
import os
import re

SPLITS_DIR = r"E:\monodepth-obstacle-warning\monodepth2\splits\kitti_subset"
OUT_DIR = r"E:\monodepth-obstacle-warning\lite-mono\splits\kitti_subset"


def convert_file(filename):
    input_path = os.path.join(SPLITS_DIR, filename)
    output_path = os.path.join(OUT_DIR, filename)
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(input_path, 'r') as f:
        lines = f.readlines()

    converted = []
    for line in lines:
        parts = line.strip().split()
        img_path = parts[0]
        # Extract: .../kitti/<date>/<date>_drive_XXXX_sync/image_0X/data/<frame>.png
        # Pattern: anything before /data/kitti is data_path
        # We need: <date>/<date>_drive_XXXX_sync
        m = re.match(r'.*?data[\\/]kitti[\\/](.+?)[\\/]image_0(\d)[\\/]data[\\/](\d+)\.(?:png|jpg)', img_path)
        if not m:
            print(f"  WARNING: could not parse: {img_path[:100]}")
            continue
        rel_path = m.group(1)  # e.g. 2011_09_26/2011_09_26_drive_0002_sync
        cam_id = m.group(2)    # 2 or 3
        frame_idx = str(int(m.group(3)))  # remove leading zeros
        side = {"2": "l", "3": "r"}[cam_id]

        converted.append(f"{rel_path} {frame_idx} {side}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(converted) + '\n')

    print(f"  {filename}: {len(converted)} entries -> {output_path}")


if __name__ == "__main__":
    for fname in ["train_files.txt", "val_files.txt", "test_files.txt"]:
        convert_file(fname)
    print("Done!")
