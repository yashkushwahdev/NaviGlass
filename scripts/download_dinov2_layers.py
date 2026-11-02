import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_BASE = "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth"

DINOV2_LAYERS_FILES = [
    "__init__.py",
    "attention.py",
    "block.py",
    "drop_path.py",
    "layer_scale.py",
    "mlp.py",
    "patch_embed.py",
    "swiglu_ffn.py",
]

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[SKIP] {dest.name} already exists")
        return
            
    print(f"[DOWNLOADING] {dest.name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=15) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"[COMPLETED] {dest.name} ({dest.stat().st_size} bytes)")

def main():
    print("=== Downloading dinov2_layers Code ===")
    for fname in DINOV2_LAYERS_FILES:
        url = f"{RAW_BASE}/depth_anything_v2/dinov2_layers/{fname}"
        dest = BASE_DIR / "depth" / "depth_anything_v2" / "dinov2_layers" / fname
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"[ERROR] Could not download {fname}: {e}")

    print("\n[SUCCESS] dinov2_layers downloaded successfully!")

if __name__ == "__main__":
    main()
