import os

root = r"E:\monodepth_project\data\kitti\2011_09_26"
sequences = [
    "2011_09_26_drive_0001_sync",
    "2011_09_26_drive_0002_sync",
    "2011_09_26_drive_0005_sync"
]

save_path = r"E:\monodepth_project\data\kitti\train_files.txt"

with open(save_path, "w", encoding="utf-8") as f:
    for seq in sequences:
        img_dir = os.path.join(root, seq, "image_02", "data")
        lidar_dir = os.path.join(root, seq, "velodyne_points", "data")

        img_files = sorted([x for x in os.listdir(img_dir) if x.endswith(".png")])
        lidar_files = sorted([x for x in os.listdir(lidar_dir) if x.endswith(".bin")])

        for img_name, lidar_name in zip(img_files, lidar_files):
            img_path = os.path.join(img_dir, img_name)
            lidar_path = os.path.join(lidar_dir, lidar_name)
            f.write(f"{img_path} {lidar_path}\n")

print("训练列表已生成：", save_path)