"""Quantitatively compare disparity predictions across checkpoints."""
import os
import sys
import numpy as np
import torch
from torchvision import transforms
import PIL.Image as pil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'monodepth2'))
import networks
from layers import disp_to_depth

CHECKPOINTS = {
    "epoch_29": r"E:\monodepth-obstacle-warning\logs\kitti_subset_v2\models\weights_29",
    "epoch_34": r"E:\monodepth-obstacle-warning\logs\kitti_subset_v2\models\weights_34",
    "epoch_39": r"E:\monodepth-obstacle-warning\logs\kitti_subset_v2\models\weights_39",
}

# Use more test images for statistical significance
def get_test_images(n=30):
    with open(r"E:\monodepth-obstacle-warning\data\kitti\test_files.txt") as f:
        paths = [l.split()[0] for l in f]
    step = max(1, len(paths) // n)
    return paths[::step][:n]


def load_model(checkpoint_dir, device):
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(os.path.join(checkpoint_dir, "encoder.pth"), map_location=device)
    feed_h, feed_w = loaded_dict_enc['height'], loaded_dict_enc['width']
    filtered = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered)
    encoder.to(device).eval()
    decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "depth.pth"), map_location=device))
    decoder.to(device).eval()
    return encoder, decoder, (feed_h, feed_w)


def predict_disp(image_path, encoder, decoder, feed_size, device):
    h, w = feed_size
    img = pil.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    tensor = transforms.ToTensor()(img.resize((w, h), pil.LANCZOS)).unsqueeze(0).to(device)
    with torch.no_grad():
        disp = decoder(encoder(tensor))[("disp", 0)]
    disp = torch.nn.functional.interpolate(disp, (orig_h, orig_w), mode="bilinear", align_corners=False)
    return disp.squeeze().cpu().numpy()


def main():
    device = torch.device("cuda")
    test_images = get_test_images(30)
    print(f"Comparing on {len(test_images)} test images")

    models = {}
    for name, ckpt in CHECKPOINTS.items():
        print(f"Loading {name}...")
        models[name] = load_model(ckpt, device)

    all_disps = {name: [] for name in CHECKPOINTS}
    for img_path in test_images:
        feed_size = (list(models.values())[0][2][0], list(models.values())[0][2][1])
        for name, (enc, dec, fs) in models.items():
            all_disps[name].append(predict_disp(img_path, enc, dec, feed_size, device))

    names = list(CHECKPOINTS.keys())

    print("\n=== Per-image Mean Absolute Difference (MAD) ===")
    for i in range(len(test_images)):
        d29 = all_disps["epoch_29"][i]
        d34 = all_disps["epoch_34"][i]
        d39 = all_disps["epoch_39"][i]
        mad_29_34 = np.mean(np.abs(d29 - d34))
        mad_34_39 = np.mean(np.abs(d34 - d39))
        mad_29_39 = np.mean(np.abs(d29 - d39))
        print(f"  img_{i:02d}: 29vs34={mad_29_34:.6f}  34vs39={mad_34_39:.6f}  29vs39={mad_29_39:.6f}")

    # Aggregate stats
    mads_29_34 = [np.mean(np.abs(all_disps["epoch_29"][i] - all_disps["epoch_34"][i])) for i in range(len(test_images))]
    mads_34_39 = [np.mean(np.abs(all_disps["epoch_34"][i] - all_disps["epoch_39"][i])) for i in range(len(test_images))]
    mads_29_39 = [np.mean(np.abs(all_disps["epoch_29"][i] - all_disps["epoch_39"][i])) for i in range(len(test_images))]

    print("\n=== Aggregate Statistics ===")
    for label, vals in [("29 vs 34", mads_29_34), ("34 vs 39", mads_34_39), ("29 vs 39", mads_29_39)]:
        print(f"  {label}: mean={np.mean(vals):.6f}, median={np.median(vals):.6f}, "
              f"min={np.min(vals):.6f}, max={np.max(vals):.6f}")

    # Per-pixel statistics: compute mean disparity and std across checkpoints for each pixel
    print("\n=== Per-pixel Dispersion Analysis ===")
    for i in range(min(5, len(test_images))):
        d29 = all_disps["epoch_29"][i]
        d34 = all_disps["epoch_34"][i]
        d39 = all_disps["epoch_39"][i]
        stacked = np.stack([d29, d34, d39], axis=0)
        mean_disp = np.mean(stacked, axis=0)
        std_disp = np.std(stacked, axis=0)
        cv = std_disp / (mean_disp + 1e-6)
        print(f"  img_{i:02d}: mean disp={np.mean(mean_disp):.4f}, "
              f"mean std={np.mean(std_disp):.6f}, mean CV={np.mean(cv):.4f}, "
              f"pixels with CV>0.1: {np.mean(cv > 0.1)*100:.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
