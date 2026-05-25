"""Compare depth predictions from multiple checkpoints on the same test images.

Generates a grid: rows = test images, cols = original + each checkpoint's depth map.
"""
import os
import sys
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Add monodepth2 to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'monodepth2'))
import networks
from layers import disp_to_depth


CHECKPOINTS = {
    "epoch_29": r"E:\monodepth-obstacle-warning\logs\kitti_subset_v2\models\weights_29",
    "epoch_34": r"E:\monodepth-obstacle-warning\logs\kitti_subset_v2\models\weights_34",
    "epoch_39": r"E:\monodepth-obstacle-warning\logs\kitti_subset_v2\models\weights_39",
}

TEST_IMAGES = [
    (r"E:\monodepth-obstacle-warning\data\kitti\2011_09_26\2011_09_26_drive_0002_sync\image_02\data\0000000000.png", "drive_0002"),
    (r"E:\monodepth-obstacle-warning\data\kitti\2011_09_26\2011_09_26_drive_0014_sync\image_02\data\0000000000.png", "drive_0014"),
    (r"E:\monodepth-obstacle-warning\data\kitti\2011_09_28\2011_09_28_drive_0045_sync\image_02\data\0000000000.png", "drive_0045"),
    (r"E:\monodepth-obstacle-warning\data\kitti\2011_09_30\2011_09_30_drive_0020_sync\image_02\data\0000000000.png", "drive_0020"),
]

OUTPUT_DIR = r"E:\monodepth-obstacle-warning\results\checkpoint_comparison"


def load_model(checkpoint_dir, device):
    encoder_path = os.path.join(checkpoint_dir, "encoder.pth")
    depth_path = os.path.join(checkpoint_dir, "depth.pth")

    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, feed_height, feed_width


def predict_disp(image_path, encoder, depth_decoder, feed_size, device):
    feed_height, feed_width = feed_size
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)
    return disp_resized.squeeze().cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all models
    models = {}
    for name, ckpt_dir in CHECKPOINTS.items():
        print(f"Loading {name} from {ckpt_dir}...")
        models[name] = load_model(ckpt_dir, device)

    # Predict for each image
    all_disps = {}  # image_label -> {epoch_name -> disp_array}
    for img_path, label in TEST_IMAGES:
        print(f"Processing {label}...")
        feed_size = (list(models.values())[0][2], list(models.values())[0][3])
        all_disps[label] = {}
        for name, (encoder, decoder, h, w) in models.items():
            all_disps[label][name] = predict_disp(img_path, encoder, decoder, (h, w), device)

    # Build comparison grid
    n_images = len(TEST_IMAGES)
    n_cols = 1 + len(CHECKPOINTS)  # original + each checkpoint
    fig, axes = plt.subplots(n_images, n_cols, figsize=(3 * n_cols, 2.5 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)

    checkpoint_names = list(CHECKPOINTS.keys())

    for row, (img_path, label) in enumerate(TEST_IMAGES):
        # Original image
        original = pil.open(img_path).convert('RGB')
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{label}\n(original)", fontsize=8)
        axes[row, 0].axis('off')

        for col, name in enumerate(checkpoint_names):
            disp = all_disps[label][name]
            vmax = np.percentile(disp, 95)
            normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped = mapper.to_rgba(disp)[:, :, :3]

            axes[row, col + 1].imshow(colormapped)
            axes[row, col + 1].set_title(f"{label}\n{name}", fontsize=8)
            axes[row, col + 1].axis('off')

    plt.tight_layout(pad=0.5)
    output_path = os.path.join(OUTPUT_DIR, "comparison_grid.png")
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved comparison grid to {output_path}")

    # Also save individual comparisons for each image (stacked vertically for easier comparison)
    for img_path, label in TEST_IMAGES:
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
        original = pil.open(img_path).convert('RGB')
        axes2[0].imshow(original)
        axes2[0].set_title("Original", fontsize=10)
        axes2[0].axis('off')

        for col, name in enumerate(checkpoint_names):
            disp = all_disps[label][name]
            vmax = np.percentile(disp, 95)
            normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped = mapper.to_rgba(disp)[:, :, :3]
            axes2[col + 1].imshow(colormapped)
            axes2[col + 1].set_title(name, fontsize=10)
            axes2[col + 1].axis('off')

        plt.tight_layout(pad=0.5)
        out = os.path.join(OUTPUT_DIR, f"{label}_comparison.png")
        fig2.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved {out}")

    plt.close('all')
    print("Done!")


if __name__ == "__main__":
    main()
