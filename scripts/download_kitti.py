"""Download KITTI raw data from public S3 bucket.
Files are publicly accessible, no login required.
Supports parallel downloads for speed.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KITTI_ROOT
import zipfile
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

ALL_DATES = {
    "2011_09_26": ["0001", "0002", "0005", "0009", "0011", "0013", "0014",
                    "0015", "0017", "0018", "0019", "0020", "0022", "0023",
                    "0027", "0028", "0029", "0032", "0035", "0036", "0039",
                    "0046", "0048", "0051", "0052", "0056", "0057", "0059",
                    "0060", "0061", "0064", "0070", "0079", "0084", "0086",
                    "0087", "0091", "0093", "0095", "0096", "0101", "0104",
                    "0106", "0113", "0117"],
    "2011_09_28": ["0001", "0002", "0016", "0021", "0034", "0035", "0037",
                    "0038", "0039", "0043", "0045", "0047", "0053", "0054",
                    "0057", "0065", "0066", "0068", "0070", "0071"],
    "2011_09_29": ["0004", "0007", "0013", "0015", "0018", "0019", "0023",
                    "0026", "0030", "0033", "0035", "0036", "0038", "0039",
                    "0040", "0042", "0045", "0047", "0049", "0052", "0055",
                    "0057", "0059", "0061", "0064", "0066", "0068", "0071",
                    "0075", "0077", "0079", "0082", "0086", "0087", "0091",
                    "0093", "0095", "0096", "0100", "0104", "0106", "0108",
                    "0110", "0113", "0117", "0119", "0121"],
    "2011_09_30": ["0016", "0018", "0020", "0027", "0028", "0033", "0034",
                    "0004", "0005", "0007", "0009", "0011", "0014", "0035"],
    "2011_10_03": ["0027", "0034", "0042", "0047", "0058", "0002", "0003",
                    "0005", "0007", "0010", "0013", "0017"],
}

_progress_lock = threading.Lock()
_total_done = 0
_total_count = 0


def download_drive(date, drive, idx):
    filename = f"{date}_drive_{drive}_sync.zip"
    folder = f"{date}_drive_{drive}_sync"
    url = f"{BASE_URL}/{date}_drive_{drive}/{filename}"

    dest_dir = os.path.join(KITTI_ROOT, date)
    os.makedirs(dest_dir, exist_ok=True)

    zip_path = os.path.join(dest_dir, filename)

    if os.path.isdir(os.path.join(dest_dir, folder)):
        img_dir = os.path.join(dest_dir, folder, "image_02", "data")
        if os.path.isdir(img_dir) and os.listdir(img_dir):
            return True, drive, "existed"

    try:
        resp = requests.get(url, stream=True, timeout=600)
        if resp.status_code != 200:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return False, drive, f"HTTP {resp.status_code}"

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)

        size_mb = downloaded / 1024 / 1024
        return True, drive, f"{size_mb:.0f}MB"

    except Exception as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False, drive, str(e)


def main():
    parser = argparse.ArgumentParser(description="Download KITTI raw data")
    parser.add_argument("--dates", nargs="+", default=["2011_09_29"],
                        help="Dates to download")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download threads (default: 4)")
    args = parser.parse_args()

    tasks = []
    for date in args.dates:
        if date not in ALL_DATES:
            print(f"{date}: unknown date, skipping")
            continue
        for drive in ALL_DATES[date]:
            tasks.append((date, drive))

    total = len(tasks)
    print(f"Downloading {', '.join(args.dates)} ({total} drives, {args.workers} parallel) to {KITTI_ROOT}\n")

    ok, failed, existed = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_drive, d, dr, i): (d, dr)
                   for i, (d, dr) in enumerate(tasks)}

        for i, future in enumerate(as_completed(futures), 1):
            success, drive, msg = future.result()
            if success:
                if msg == "existed":
                    existed += 1
                    print(f"[{i}/{total}] {drive}: skipped (exists)")
                else:
                    ok += 1
                    print(f"[{i}/{total}] {drive}: done ({msg})")
            else:
                failed += 1
                print(f"[{i}/{total}] {drive}: FAILED ({msg})")

    print(f"\nDone: {ok} ok, {existed} existed, {failed} failed (total {total})")


if __name__ == "__main__":
    main()
