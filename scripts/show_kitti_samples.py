import os
import cv2
import matplotlib.pyplot as plt

img_dir = r"E:\monodepth_project\data\kitti\2011_09_26\2011_09_26_drive_0001_sync\image_02\data"
save_dir = r"E:\monodepth_project\results\plots"
os.makedirs(save_dir, exist_ok=True)

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
sample_indices = [0, 10, 20, 50]

plt.figure(figsize=(16, 8))

for j, i in enumerate(sample_indices):
    img_path = os.path.join(img_dir, img_files[i])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, j + 1)
    plt.imshow(img)
    plt.title(f"Sample {i}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "sample_images_grid.png"))
plt.show()