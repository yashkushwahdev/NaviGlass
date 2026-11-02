"""Download official Piper TTS voice model and configuration (en_US-amy-medium).

Voice:
    en_US-amy-medium (Female, US English)
Source:
    https://huggingface.co/rhasspy/piper-voices
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

VOICE_ONNX_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
)
VOICE_JSON_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
)


def download_file(url: str, dest_path: Path, expected_min_size: int = 1000) -> bool:
    """Downloads a file if not already present or incomplete."""
    if dest_path.exists() and dest_path.stat().st_size >= expected_min_size:
        print(f"[EXISTS] {dest_path.name} already downloaded ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    print(f"[DOWNLOADING] {dest_path.name} from {url}...")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "NaviGlass-Downloader/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as f:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            last_print = time.time()

            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if time.time() - last_print > 1.5:
                    last_print = time.time()
                    if total_size > 0:
                        pct = downloaded / total_size * 100.0
                        mb = downloaded / (1024 * 1024)
                        print(f"  -> {mb:.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")

        temp_path.replace(dest_path)
        elapsed = time.time() - t0
        print(f"[SUCCESS] Downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB in {elapsed:.1f}s)")
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {dest_path.name}: {e}") from e


def verify_voice_files(model_path: Path, config_path: Path) -> bool:
    """Verifies that the ONNX model and JSON config exist and are structurally valid."""
    if not model_path.exists() or model_path.stat().st_size < 1_000_000:
        raise ValueError(f"Model file missing or corrupted: {model_path}")

    if not config_path.exists() or config_path.stat().st_size < 100:
        raise ValueError(f"Config file missing or corrupted: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "audio" not in data or "sample_rate" not in data["audio"]:
                raise ValueError("JSON configuration is missing audio/sample_rate settings.")
            sample_rate = data["audio"]["sample_rate"]
            print(f"[VERIFIED] JSON config valid: sample_rate={sample_rate} Hz, phonemizer={data.get('phonemizer', 'default')}")
    except Exception as e:
        raise ValueError(f"Failed to parse voice configuration JSON: {e}") from e

    print(f"[VERIFIED] Voice model verified successfully: {model_path.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Piper TTS en_US-amy-medium voice model.")
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="models/voices",
        help="Destination directory for voice models",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir).resolve()
    model_path = dest_dir / "en_US-amy-medium.onnx"
    config_path = dest_dir / "en_US-amy-medium.onnx.json"

    print("=" * 60)
    print("NaviGlass Piper Voice Setup: en_US-amy-medium")
    print("=" * 60)
    print(f"Target Directory: {dest_dir}")

    # 1. Download ONNX model (~63 MB)
    download_file(VOICE_ONNX_URL, model_path, expected_min_size=50_000_000)

    # 2. Download JSON config (~5 KB)
    download_file(VOICE_JSON_URL, config_path, expected_min_size=1_000)

    # 3. Verify files
    verify_voice_files(model_path, config_path)
    print("=" * 60)
    print("Piper voice model ready for NaviGlass.")
    print("=" * 60)


if __name__ == "__main__":
    main()

