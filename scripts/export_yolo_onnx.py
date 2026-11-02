"""Export local YOLOv8 PyTorch model to ONNX for Raspberry Pi 4 CPU inference.

Usage:
    python scripts/export_yolo_onnx.py [--weights models/yolov8n.pt] [--imgsz 320] [--output models/yolov8n.onnx]

Requirements on export machine (Windows dev machine):
    - ultralytics
    - torch
    - onnx
    - onnxruntime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_yolo_to_onnx(
    weights_path: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 320,
    opset: int = 17,
) -> Path:
    """Exports a local YOLO PyTorch model to ONNX with static shapes for Raspberry Pi 4 CPU.

    Args:
        weights_path: Path to existing .pt model file.
        output_path: Destination path for .onnx model file (optional).
        imgsz: Image resolution (static square dimension, e.g. 320).
        opset: ONNX opset version (default 17).

    Returns:
        Path to the exported .onnx model.
    """
    weights_file = Path(weights_path).resolve()
    if not weights_file.exists():
        raise FileNotFoundError(f"Input weights file not found: {weights_file}")

    if weights_file.suffix.lower() != ".pt":
        raise ValueError(f"Input model must be a PyTorch .pt file, got: {weights_file.name}")

    orig_stat = weights_file.stat()
    orig_mtime = orig_stat.st_mtime
    orig_size = orig_stat.st_size

    print("=" * 60)
    print("NaviGlass YOLO -> ONNX Exporter for Raspberry Pi 4")
    print("=" * 60)
    print(f"Source PyTorch weights: {weights_file} ({orig_size / (1024*1024):.2f} MB)")
    print(f"Target inference size:  {imgsz}x{imgsz} (static dimensions)")
    print(f"Target ONNX opset:     {opset}")

    # Verify Ultralytics & PyTorch
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics is required for export. Install with: pip install ultralytics") from e

    try:
        import onnx
    except ImportError as e:
        raise RuntimeError("onnx is required for export. Install with: pip install onnx") from e

    # Load local model
    print("\n[1/4] Loading local model weights...")
    try:
        model = YOLO(str(weights_file))
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from {weights_file}: {e}") from e

    # Verify input weights were not modified or overwritten
    if weights_file.stat().st_size != orig_size:
        raise RuntimeError("Integrity error: Source weights file was modified during load.")

    # Export to ONNX
    print(f"\n[2/4] Exporting to ONNX format (imgsz={imgsz}, dynamic=False, opset={opset})...")
    try:
        exported_path_str = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            opset=opset,
            simplify=True,
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}") from e

    exported_file = Path(exported_path_str).resolve()
    if not exported_file.exists():
        raise RuntimeError(f"Export completed without error but exported file was not found at {exported_file}")

    # Handle custom destination if specified
    if output_path is not None:
        target_dest = Path(output_path).resolve()
        target_dest.parent.mkdir(parents=True, exist_ok=True)
        if target_dest != exported_file:
            import shutil
            shutil.copy2(exported_file, target_dest)
            exported_file = target_dest
            print(f"Copied exported model to requested destination: {exported_file}")

    # Confirm original file was never overwritten
    curr_stat = weights_file.stat()
    if curr_stat.st_size != orig_size or curr_stat.st_mtime != orig_mtime:
        raise RuntimeError("Safety check failed: Original PyTorch weights file was modified!")

    print(f"\n[3/4] Export succeeded: {exported_file} ({exported_file.stat().st_size / (1024*1024):.2f} MB)")

    # Verify exported model using onnx and onnxruntime
    print("\n[4/4] Validating exported ONNX graph & tensor signatures...")
    try:
        onnx_model = onnx.load(str(exported_file))
        onnx.checker.check_model(onnx_model)
        print(" -> ONNX graph check passed.")
    except Exception as e:
        raise RuntimeError(f"Exported ONNX model graph validation failed: {e}") from e

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(exported_file), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print(" -> Model Input Signature:")
        for inp in inputs:
            print(f"      Name:  {inp.name}")
            print(f"      Shape: {inp.shape}")
            print(f"      Type:  {inp.type}")

        print(" -> Model Output Signature:")
        for out in outputs:
            print(f"      Name:  {out.name}")
            print(f"      Shape: {out.shape}")
            print(f"      Type:  {out.type}")

        meta = session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            print(" -> Class names found in model metadata.")

    except ImportError:
        print(" -> Warning: onnxruntime not available to inspect session signatures.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize onnxruntime test session: {e}") from e

    print("=" * 60)
    print(f"SUCCESS: YOLO ONNX model is ready at: {exported_file}")
    print("=" * 60)
    return exported_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 PyTorch model to ONNX for Raspberry Pi 4 CPU.")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolov8n.pt",
        help="Path to input PyTorch weights file (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/yolov8n.onnx",
        help="Path to output ONNX file (.onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image resolution (default: 320 for RPi4 CPU)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    try:
        export_yolo_to_onnx(
            weights_path=args.weights,
            output_path=args.output,
            imgsz=args.imgsz,
            opset=args.opset,
        )
    except Exception as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

