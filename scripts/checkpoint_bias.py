"""Analyze bias direction between checkpoints."""
import os, sys
import numpy as np
import torch
from torchvision import transforms
import PIL.Image as pil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'monodepth2'))
import networks

names = ['epoch_29', 'epoch_34', 'epoch_39']
base = r'E:\monodepth-obstacle-warning\logs\kitti_subset_v2\models'

with open(r'E:\monodepth-obstacle-warning\data\kitti\test_files.txt') as f:
    paths = [l.split()[0] for l in f]
paths = paths[::57][:30]

device = torch.device('cuda')
models = {}
for name in names:
    ckpt = os.path.join(base, f'weights_{name.split("_")[1]}')
    enc = networks.ResnetEncoder(18, False)
    d = torch.load(os.path.join(ckpt, 'encoder.pth'), map_location=device)
    fh, fw = d['height'], d['width']
    filt = {k: v for k, v in d.items() if k in enc.state_dict()}
    enc.load_state_dict(filt)
    enc.to(device).eval()
    dec = networks.DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=range(4))
    dec.load_state_dict(torch.load(os.path.join(ckpt, 'depth.pth'), map_location=device))
    dec.to(device).eval()
    models[name] = (enc, dec, (fh, fw))

all_means = {n: [] for n in names}
edge_sharpness = {n: [] for n in names}
for img_path in paths:
    img = pil.open(img_path).convert('RGB')
    ow, oh = img.size
    h, w = models['epoch_29'][2]
    tensor = transforms.ToTensor()(img.resize((w, h), pil.LANCZOS)).unsqueeze(0).to(device)
    for name, (enc, dec, _) in models.items():
        with torch.no_grad():
            disp = dec(enc(tensor))[('disp', 0)]
        disp = torch.nn.functional.interpolate(disp, (oh, ow), mode='bilinear', align_corners=False)
        disp_np = disp.squeeze().cpu().numpy()
        all_means[name].append(np.mean(disp_np))
        # Edge sharpness: mean absolute laplacian
        lap = np.abs(disp_np[1:-1, 1:-1] * 4 - disp_np[:-2, 1:-1] - disp_np[2:, 1:-1] - disp_np[1:-1, :-2] - disp_np[1:-1, 2:])
        edge_sharpness[name].append(np.mean(lap))

means_29 = np.array(all_means['epoch_29'])
means_34 = np.array(all_means['epoch_34'])
means_39 = np.array(all_means['epoch_39'])

print('=== Mean disparity ===')
for name in names:
    print(f'  {name}: {np.mean(all_means[name]):.6f}')

print(f'\n=== Bias direction ===')
print(f'  34 - 29: {np.mean(means_34 - means_29):.6f}  ({(means_34 > means_29).mean()*100:.0f}% images: 34 > 29)')
print(f'  39 - 34: {np.mean(means_39 - means_34):.6f}  ({(means_39 > means_34).mean()*100:.0f}% images: 39 > 34)')
print(f'  39 - 29: {np.mean(means_39 - means_29):.6f}  ({(means_39 > means_29).mean()*100:.0f}% images: 39 > 29)')

print(f'\n=== Edge sharpness (mean |laplacian|, higher = more detail) ===')
for name in names:
    print(f'  {name}: {np.mean(edge_sharpness[name]):.6f}')
print(f'  Ratios: 34/29={np.mean(edge_sharpness["epoch_34"])/np.mean(edge_sharpness["epoch_29"]):.4f}, '
      f'39/29={np.mean(edge_sharpness["epoch_39"])/np.mean(edge_sharpness["epoch_29"]):.4f}')
