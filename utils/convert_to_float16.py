"""
Convert NEMO Scan model weights from float32 to float16.

Loads every .pth file from weights/lung/, converts all floating-point
tensors in the state_dict to float16 (non-float buffers such as
BatchNorm's num_batches_tracked are left as-is), and writes the
converted checkpoint to weights/lung/float16/ with identical filenames.

Original float32 weights are never modified.

Usage:
    python utils/convert_to_float16.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

SRC_DIR  = ROOT / "weights" / "lung"
DST_DIR  = ROOT / "weights" / "lung" / "float16"


def convert_checkpoint(src_path: Path, dst_path: Path) -> tuple[int, int]:
    """
    Converts all floating-point tensors in a checkpoint state_dict to float16.
    Returns (original_bytes, converted_bytes).
    """
    checkpoint = torch.load(src_path, map_location="cpu", weights_only=False)

    if "state_dict" not in checkpoint:
        raise KeyError(f"No 'state_dict' key found in {src_path.name}. "
                       f"Available keys: {list(checkpoint.keys())}")

    original_sd = checkpoint["state_dict"]
    converted_sd = {
        k: v.half() if v.is_floating_point() else v
        for k, v in original_sd.items()
    }
    checkpoint["state_dict"] = converted_sd

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dst_path)

    return src_path.stat().st_size, dst_path.stat().st_size


def fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1_048_576:.1f} MB"


def main():
    if not SRC_DIR.exists():
        print(f"ERROR: Source directory not found: {SRC_DIR}")
        sys.exit(1)

    pth_files = sorted(p for p in SRC_DIR.iterdir()
                       if p.suffix == ".pth" and p.parent == SRC_DIR)

    if not pth_files:
        print(f"No .pth files found in {SRC_DIR}")
        sys.exit(1)

    print(f"Source : {SRC_DIR}")
    print(f"Output : {DST_DIR}")
    print(f"Models : {len(pth_files)}")
    print("-" * 60)

    total_src = 0
    total_dst = 0

    for src in pth_files:
        dst = DST_DIR / src.name
        print(f"  Converting  {src.name} ...", end="", flush=True)
        try:
            src_bytes, dst_bytes = convert_checkpoint(src, dst)
            saving_pct = (1 - dst_bytes / src_bytes) * 100
            print(f"  {fmt_mb(src_bytes):>10}  ->  {fmt_mb(dst_bytes):>9}  "
                  f"({saving_pct:.0f}% smaller)")
            total_src += src_bytes
            total_dst += dst_bytes
        except Exception as exc:
            print(f"\n    ERROR: {exc}")

    print("-" * 60)
    total_saving = (1 - total_dst / total_src) * 100 if total_src else 0
    print(f"  TOTAL        {fmt_mb(total_src):>10}  ->  {fmt_mb(total_dst):>9}  "
          f"({total_saving:.0f}% smaller)")
    print(f"\nDone. Float16 weights saved to:\n  {DST_DIR}")


if __name__ == "__main__":
    main()
