"""Build an obstacle-warning demo video from frames and disparity maps."""

import argparse
import json
import os
import sys

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from obstacle_warning_image import (  # noqa: E402
    compute_warning,
    default_disp_path,
    draw_roi,
    draw_warning_panel,
    load_disparity,
    make_comparison,
    normalize_disparity,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a video with input, disparity, and obstacle warning panels."
    )
    parser.add_argument("--frames_dir", required=True, help="Directory containing frame images.")
    parser.add_argument("--output_path", required=True, help="Output mp4 path.")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"])
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--roi_x1", type=float, default=0.33)
    parser.add_argument("--roi_x2", type=float, default=0.67)
    parser.add_argument("--roi_y1", type=float, default=0.40)
    parser.add_argument("--roi_y2", type=float, default=0.78)
    parser.add_argument("--near_percentile", type=float, default=88.0)
    parser.add_argument("--caution_ratio", type=float, default=0.08)
    parser.add_argument("--danger_ratio", type=float, default=0.18)
    parser.add_argument(
        "--layout",
        default="overlay",
        choices=["overlay", "triple"],
        help="overlay keeps a 16:9-friendly frame; triple shows input/depth/warning side by side.",
    )
    parser.add_argument(
        "--inset_scale",
        type=float,
        default=0.32,
        help="Depth inset width as a fraction of frame width in overlay layout.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional limit for a short preview render.",
    )
    return parser.parse_args()


def load_fps(frames_dir, fallback=30.0):
    metadata_path = os.path.join(frames_dir, "frames_metadata.json")
    if not os.path.exists(metadata_path):
        return fallback

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return float(metadata.get("output_fps") or fallback)


def list_frames(frames_dir, ext):
    frames = []
    suffix = f".{ext.lower()}"
    for name in os.listdir(frames_dir):
        if not name.lower().endswith(suffix):
            continue
        if name.lower().endswith("_disp.jpeg") or name.lower().endswith("_warning.jpg"):
            continue
        frames.append(os.path.join(frames_dir, name))
    return sorted(frames)


def make_overlay(annotated, disp_vis, inset_scale):
    output = annotated.copy()
    height, width = output.shape[:2]
    inset_width = max(160, int(width * inset_scale))
    inset_height = max(90, int(inset_width * height / width))
    disp_inset = cv2.resize(disp_vis, (inset_width, inset_height), interpolation=cv2.INTER_LINEAR)

    pad = max(12, int(width * 0.018))
    x1 = width - inset_width - pad
    y1 = height - inset_height - pad
    x2 = width - pad
    y2 = height - pad

    cv2.rectangle(output, (x1 - 4, y1 - 32), (x2 + 4, y2 + 4), (255, 255, 255), thickness=-1)
    cv2.rectangle(output, (x1 - 4, y1 - 32), (x2 + 4, y2 + 4), (40, 40, 40), thickness=2)
    cv2.putText(
        output,
        "Predicted Disparity",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    output[y1:y2, x1:x2] = disp_inset
    return output


def annotate_frame(frame_path, args):
    image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read frame: {frame_path}")

    height, width = image.shape[:2]
    disp_path = default_disp_path(frame_path)
    disp = load_disparity(disp_path, width, height)
    disp_norm = normalize_disparity(disp)
    disp_vis = cv2.applyColorMap(disp_norm, cv2.COLORMAP_MAGMA)

    roi = (
        int(width * args.roi_x1),
        int(height * args.roi_y1),
        int(width * args.roi_x2),
        int(height * args.roi_y2),
    )
    status, near_ratio, mean_disp, near_threshold = compute_warning(
        disp,
        roi,
        args.near_percentile,
        args.caution_ratio,
        args.danger_ratio,
    )

    annotated = image.copy()
    draw_roi(annotated, roi, status)
    draw_warning_panel(annotated, status, near_ratio, mean_disp, near_threshold)
    if args.layout == "triple":
        comparison = make_comparison(image.copy(), disp_vis, annotated)
    else:
        comparison = make_overlay(annotated, disp_vis, args.inset_scale)
    return comparison, status


def main():
    args = parse_args()
    frames = list_frames(args.frames_dir, args.ext)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise FileNotFoundError(f"No .{args.ext} frames found in {args.frames_dir}")

    fps = args.fps or load_fps(args.frames_dir)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    first_frame, first_status = annotate_frame(frames[0], args)
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.output_path}")

    counts = {first_status: 1}
    writer.write(first_frame)
    for index, frame_path in enumerate(frames[1:], start=2):
        frame, status = annotate_frame(frame_path, args)
        counts[status] = counts.get(status, 0) + 1
        writer.write(frame)
        if index % 30 == 0:
            print(f"processed={index}/{len(frames)}")

    writer.release()

    print(f"frames={len(frames)}")
    print(f"fps={fps:.2f}")
    print(f"status_counts={counts}")
    print(f"saved={args.output_path}")


if __name__ == "__main__":
    main()
