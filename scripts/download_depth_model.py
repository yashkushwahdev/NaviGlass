import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
DEST_PATH = MODELS_DIR / "depth_anything_v2_metric_hypersim_vits.pth"

RAW_BASE = "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth"
CODE_FILES = {
    "depth/depth_anything_v2/__init__.py": f"{RAW_BASE}/depth_anything_v2/__init__.py",
    "depth/depth_anything_v2/dpt.py": f"{RAW_BASE}/depth_anything_v2/dpt.py",
    "depth/depth_anything_v2/dinov2.py": f"{RAW_BASE}/depth_anything_v2/dinov2.py",
    "depth/depth_anything_v2/util/__init__.py": f"{RAW_BASE}/depth_anything_v2/util/__init__.py",
    "depth/depth_anything_v2/util/blocks.py": f"{RAW_BASE}/depth_anything_v2/util/blocks.py",
    "depth/depth_anything_v2/util/transform.py": f"{RAW_BASE}/depth_anything_v2/util/transform.py",
}

def download_file(url: str, dest: Path, is_binary: bool = False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if is_binary and dest.stat().st_size > 50_000_000:
            print(f"[SKIP] {dest.name} already exists ({dest.stat().st_size / (1024*1024):.1f} MB)")
            return
        elif not is_binary:
            print(f"[SKIP] {dest.name} already exists")
            return
            
    print(f"[DOWNLOADING] {dest.name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    t0 = time.time()
    
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        last_log = time.time()
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_log > 3.0:
                last_log = time.time()
                pct = (downloaded / total * 100) if total else 0
                print(f" -> {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({pct:.1f}%)")
                
    elapsed = time.time() - t0
    print(f"[COMPLETED] {dest.name}: {dest.stat().st_size / (1024*1024):.2f} MB in {elapsed:.1f}s")

def main():
    print("=== Downloading Depth Anything V2 Metric Architecture Code ===")
    for rel_path, url in CODE_FILES.items():
        try:
            download_file(url, BASE_DIR / rel_path, is_binary=False)
        except Exception as e:
            print(f"[WARN] Could not download {rel_path} directly ({e}), checking alternatives...")

    print("\n=== Downloading Small Indoor Metric Weights ===")
    try:
        download_file(WEIGHTS_URL, DEST_PATH, is_binary=True)
    except Exception as e:
        print(f"[ERROR] Failed to download weights from {WEIGHTS_URL}: {e}")
        sys.exit(1)
        
    print("\n[SUCCESS] Depth Anything V2 Metric Small Indoor model downloaded successfully!")

if __name__ == "__main__":
    main()
