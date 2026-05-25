"""
Export TensorBoard event scalars to CSV and generate loss/curve plots.
Uses streaming parser to efficiently handle large event files.
"""
import sys
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

PROJECT_ROOT = Path("E:/monodepth-obstacle-warning")
RESULTS = PROJECT_ROOT / "results"

# V1: final complete-run event file only (May 8, largest file)
V1_TRAIN = "logs/kitti_subset_model/train/events.out.tfevents.1778230285.LAPTOP-CL0N41BD"
V1_VAL   = "logs/kitti_subset_model/val/events.out.tfevents.1778230285.LAPTOP-CL0N41BD"
# V2
V2_TRAIN = "logs/kitti_subset_v2/train/events.out.tfevents.1778334326.LAPTOP-CL0N41BD"
V2_VAL   = "logs/kitti_subset_v2/val/events.out.tfevents.1778334326.LAPTOP-CL0N41BD"


def load_scalars(event_file: Path) -> dict[str, list[tuple[int, float]]]:
    """Stream-read scalars from a TensorBoard event file, skipping images.

    Returns {tag: [(step, value), ...]} sorted by step.
    """
    loader = LegacyEventFileLoader(str(event_file))
    data = defaultdict(list)
    count = 0
    for event in loader.Load():
        count += 1
        if not event.HasField("summary"):
            continue
        for value in event.summary.value:
            if value.HasField("simple_value"):
                data[value.tag].append((event.step, value.simple_value))
        if count % 50000 == 0:
            print(f"    ... processed {count} events, {sum(len(v) for v in data.values())} scalars so far")
    print(f"    Total events scanned: {count}")
    return {tag: sorted(pts, key=lambda x: x[0]) for tag, pts in data.items()}


def smooth(data: list[float], window: int = 50) -> list[float]:
    """Simple moving-average smoother."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid").tolist()


def export_csv(tag_to_points: dict, out_path: Path, smooth_window: int = 50):
    """Write CSV: tag, step, value, smoothed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, pts in sorted(tag_to_points.items()):
        steps, values = zip(*pts) if pts else ([], [])
        smoothed = smooth(list(values), smooth_window)
        offset = len(values) - len(smoothed)
        for i in range(len(values)):
            s_val = smoothed[i - offset] if i >= offset else ""
            rows.append((tag, steps[i], values[i], s_val))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "value", "smoothed"])
        w.writerows(rows)


def make_loss_plot(train_data: dict, val_data: dict, out_path: Path):
    """Plot training+validation loss curves."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: total loss
    ax = axes[0]
    t_pts = train_data.get("loss", [])
    v_pts = val_data.get("loss", [])
    if t_pts:
        t_steps, t_vals = zip(*t_pts)
        ax.plot(t_steps, t_vals, alpha=0.15, color="tab:blue", linewidth=0.5)
        s = smooth(list(t_vals), 50)
        ax.plot(t_steps[-len(s):], s, color="tab:blue", label="Train loss")
    if v_pts:
        v_steps, v_vals = zip(*v_pts)
        s = smooth(list(v_vals), 10)
        ax.plot(v_steps[-len(s):], s, color="tab:orange", label="Val loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: per-scale losses
    ax = axes[1]
    for scl in range(4):
        tag = f"loss/{scl}"
        pts = train_data.get(tag, [])
        if pts:
            steps, vals = zip(*pts)
            s = smooth(list(vals), 50)
            ax.plot(steps[-len(s):], s, label=f"Scale {scl}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Per-Scale Training Loss (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    tasks = [
        ("v1_baseline", PROJECT_ROOT / V1_TRAIN, PROJECT_ROOT / V1_VAL),
        ("v2_baseline", PROJECT_ROOT / V2_TRAIN, PROJECT_ROOT / V2_VAL),
    ]

    for name, train_file, val_file in tasks:
        print(f"\n{'='*60}")
        print(f"Processing {name} ...")
        print(f"  Train: {train_file}  ({train_file.stat().st_size / 1e6:.1f} MB)")
        print(f"  Val:   {val_file}  ({val_file.stat().st_size / 1e6:.1f} MB)")

        print("  Loading training scalars ...")
        train_data = load_scalars(train_file)
        for tag, pts in train_data.items():
            print(f"    {tag}: {len(pts)} points")

        print("  Loading validation scalars ...")
        val_data = load_scalars(val_file)
        for tag, pts in val_data.items():
            print(f"    {tag}: {len(pts)} points")

        # Merge train+val
        merged = defaultdict(list)
        for tag, pts in train_data.items():
            merged[tag].extend(pts)
        for tag, pts in val_data.items():
            merged[tag].extend(pts)

        out_dir = RESULTS / name
        export_csv(dict(merged), out_dir / "metrics.csv")
        print(f"  Wrote {out_dir / 'metrics.csv'}")

        # loss_curve.csv
        loss_csv = out_dir / "loss_curve.csv"
        t_pts = train_data.get("loss", [])
        v_pts = val_data.get("loss", [])
        t_dict = {s: v for s, v in t_pts}
        v_dict = {s: v for s, v in v_pts}
        all_steps = sorted(set(t_dict) | set(v_dict))
        t_raw = [t_dict[s] for s in all_steps]
        v_raw = [v_dict.get(s) for s in all_steps]
        t_smooth = smooth(t_raw, 50)
        v_vals_only = [x for x in v_raw if x is not None]
        v_smooth = smooth(v_vals_only, 5)

        with open(loss_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss_raw", "train_loss_smooth", "val_loss_raw", "val_loss_smooth"])
            vs_idx = 0
            for i, s in enumerate(all_steps):
                tv = t_raw[i]
                vv = v_raw[i]
                ts = t_smooth[i - (len(t_raw) - len(t_smooth))] if i >= (len(t_raw) - len(t_smooth)) else ""
                vs = ""
                if vv is not None:
                    vs_idx += 1
                    if vs_idx - 1 < len(v_smooth):
                        vs = v_smooth[vs_idx - 1]
                w.writerow([s, tv, ts, vv if vv is not None else "", vs])
        print(f"  Wrote {loss_csv}")

        make_loss_plot(train_data, val_data, out_dir / "loss_curve.png")
        print(f"  Wrote {out_dir / 'loss_curve.png'}")

    print(f"\nDone! Output in: {RESULTS}")


if __name__ == "__main__":
    main()
