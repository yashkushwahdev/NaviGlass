"""Cross-platform Audio and Speech Output diagnostic utility for Raspberry Pi 4 & Desktop.

Tests speech synthesis using Linux espeak-ng/espeak or Windows SAPI5.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from naviglass.audio import WindowsTTS
from naviglass.config import AudioConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="NaviGlass Speech Output Diagnostic")
    parser.add_argument("--rate", type=int, default=180, help="Speech rate in wpm (default: 180)")
    parser.add_argument("--phrase", type=str, default="NaviGlass speech diagnostic complete.", help="Test phrase")
    args = parser.parse_args()

    print("==================================================")
    print("        NaviGlass Speech Output Diagnostic        ")
    print("==================================================")
    print(f"Speech Rate: {args.rate} wpm")
    print(f'Utterance:   "{args.phrase}"')
    print("--------------------------------------------------")

    cfg = AudioConfig(speech_rate_wpm=args.rate)
    tts = WindowsTTS(cfg)
    tts.start()

    print(f"Active Backend: {getattr(tts, '_backend_type', 'unknown')}")
    print(f"TTS Available:  {tts.is_available}")

    success = tts.speak_async(args.phrase)
    if not success:
        print("[ERROR] Failed to enqueue speech utterance.", file=sys.stderr)
        tts.stop()
        sys.exit(1)

    tts.wait_until_done(timeout=6.0)
    time.sleep(0.6)
    tts.stop()

    print("--------------------------------------------------")
    print("Speech diagnostic PASSED.")


if __name__ == "__main__":
    main()
