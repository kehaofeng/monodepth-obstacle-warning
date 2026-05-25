"""Generate gt_depths.npz for a KITTI subset split from velodyne data."""
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "monodepth2"))
from kitti_utils import generate_depth_map

DATA_PATH = "E:/monodepth-obstacle-warning/data/kitti"
SPLITS_DIR = "E:/monodepth-obstacle-warning/monodepth2/splits/kitti_subset"

test_files = os.path.join(SPLITS_DIR, "test_files.txt")
gt_output = os.path.join(SPLITS_DIR, "gt_depths.npz")

with open(test_files, "r") as f:
    lines = f.readlines()

depths = []
errors = []

for idx, line in enumerate(tqdm(lines, desc="Generating gt depths")):
    parts = line.strip().split()
    folder = parts[0]              # e.g. 2011_09_26/2011_09_26_drive_0002_sync
    frame_index = int(parts[1])    # e.g. 1
    side = parts[2]                # e.g. l
    cam = 2 if side == "l" else 3

    calib_dir = os.path.join(DATA_PATH, folder.split("/")[0])
    velo_file = os.path.join(DATA_PATH, folder,
                             "velodyne_points/data/{:010d}.bin".format(frame_index))

    if not os.path.isfile(velo_file):
        errors.append((idx, folder, frame_index, "missing velodyne"))
        depths.append(np.zeros((375, 1242), dtype=np.float32))
        continue

    try:
        depth = generate_depth_map(calib_dir, velo_file, cam)
        depths.append(depth)
    except Exception as e:
        errors.append((idx, folder, frame_index, str(e)))
        depths.append(np.zeros((375, 1242), dtype=np.float32))

if errors:
    print(f"WARNING: {len(errors)} files had errors:")
    for e in errors[:5]:
        print(f"  idx={e[0]} {e[1]} frame={e[2]}: {e[3]}")
    if len(errors) > 5:
        print(f"  ... and {len(errors)-5} more")

print(f"Saving {len(depths)} depth maps to {gt_output}")
np.savez_compressed(gt_output, data=np.array(depths, dtype=object))
print("Done!")
