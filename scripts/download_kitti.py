"""Download additional KITTI raw data from public S3 bucket.

Files are publicly accessible, no login required.
"""

import os
import zipfile
import requests

DATE = "2011_09_28"
KITTI_ROOT = r"E:\monodepth-obstacle-warning\data\kitti"
BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"

# Correct drive list for 2011_09_28 (from official downloader)
# Picking ~20 drives for a reasonable dataset size
DRIVES = [
    "0001", "0002", "0016", "0021", "0034", "0035", "0037", "0038",
    "0039", "0043", "0045", "0047", "0053", "0054", "0057", "0065",
    "0066", "0068", "0070", "0071",
]


def download_drive(date, drive):
    filename = f"{date}_drive_{drive}_sync.zip"
    folder = f"{date}_drive_{drive}_sync"
    url = f"{BASE_URL}/{date}_drive_{drive}/{filename}"

    dest_dir = os.path.join(KITTI_ROOT, date)
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.isdir(os.path.join(dest_dir, folder)):
        print(f"  {drive}: 已存在，跳过")
        return True

    zip_path = os.path.join(dest_dir, filename)
    print(f"  {drive}: 下载中...", end=" ", flush=True)

    try:
        resp = requests.get(url, stream=True, timeout=300)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code}")
            return False

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {drive}: {downloaded/total*100:.0f}%", end="", flush=True)

        size_mb = downloaded / 1024 / 1024
        print(f"\r  {drive}: 下载完成 ({size_mb:.0f}MB)，解压中...", end=" ", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
        print("完成")
        return True

    except Exception as e:
        print(f"失败: {e}")
        return False


def main():
    print(f"下载 {DATE} ({len(DRIVES)} 个序列)\n")

    ok = sum(1 for d in DRIVES if download_drive(DATE, d))
    print(f"\n成功: {ok}/{len(DRIVES)}")


if __name__ == "__main__":
    main()
