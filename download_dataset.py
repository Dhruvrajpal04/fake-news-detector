"""
Dataset downloader for the Fake News Detection System.
Downloads the official WELFake dataset (Word Embedding over Linguistic Features
for Fake News Detection) from Zenodo (Record 4561253).
Dataset License: Creative Commons Attribution 4.0 International (CC BY 4.0).
Citation: P. K. Verma, P. Agrawal, I. Amorim and R. Prodan,
"WELFake: Word Embedding Over Linguistic Features for Fake News Detection,"
IEEE Transactions on Computational Social Systems, doi: 10.1109/TCSS.2021.3068519.
"""

import os
import sys
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_PATH = os.path.join(DATA_DIR, "WELFake_Dataset.csv")
ZENODO_URL = "https://zenodo.org/records/4561253/files/WELFake_Dataset.csv"
EXPECTED_SIZE = 245086152  # ~233.7 MB


def download_welfake_dataset(force: bool = False) -> str:
    """
    Downloads the WELFake dataset from Zenodo to data/WELFake_Dataset.csv.
    Streams chunks and shows progress.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATASET_PATH) and not force:
        size = os.path.getsize(DATASET_PATH)
        if size > 100_000:  # Valid non-empty file
            print(f"[OK] WELFake dataset already exists at: {DATASET_PATH} ({size / (1024*1024):.2f} MB)")
            return DATASET_PATH
        else:
            print("[WARN] Existing file appears corrupted or incomplete. Re-downloading...")

    print("=" * 70)
    print("Downloading WELFake dataset from official Zenodo repository...")
    print(f"Source URL: {ZENODO_URL}")
    print(f"Destination: {DATASET_PATH}")
    print("=" * 70)

    part_path = DATASET_PATH + ".part"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FakeNewsDetector/1.0"}

    try:
        response = requests.get(ZENODO_URL, stream=True, headers=headers, timeout=60)
        response.raise_for_status()

        total_length = response.headers.get("content-length")
        total_bytes = int(total_length) if total_length else EXPECTED_SIZE
        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB chunks

        with open(part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    done_mb = downloaded / (1024 * 1024)
                    tot_mb = total_bytes / (1024 * 1024)
                    percent = (downloaded / total_bytes) * 100 if total_bytes else 0
                    sys.stdout.write(f"\rDownloading: {done_mb:.1f} MB / {tot_mb:.1f} MB ({percent:.1f}%)")
                    sys.stdout.flush()

        print("\nDownload complete! Finalizing file...")
        if os.path.exists(DATASET_PATH):
            os.remove(DATASET_PATH)
        os.rename(part_path, DATASET_PATH)
        print(f"[SUCCESS] Dataset saved to {DATASET_PATH} ({os.path.getsize(DATASET_PATH) / (1024*1024):.2f} MB)")
        return DATASET_PATH

    except Exception as e:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        print(f"\n[ERROR] Failed to download dataset: {e}")
        print("\nManual Download Instructions:")
        print("1. Visit https://zenodo.org/records/4561253")
        print("2. Download 'WELFake_Dataset.csv' (245 MB)")
        print(f"3. Place the file in: {DATASET_PATH}")
        raise


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    try:
        download_welfake_dataset(force=force_download)
    except Exception:
        sys.exit(1)
