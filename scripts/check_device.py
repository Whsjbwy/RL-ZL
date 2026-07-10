"""Report Python/PyTorch/CUDA status without modifying the environment."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed in this interpreter")
        print("Stage 0 uses NumPy and can still be validated; install PyTorch before SAC training.")
        return 0

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {properties.name}; "
            f"VRAM={properties.total_memory / 1024**3:.2f} GiB; "
            f"compute={properties.major}.{properties.minor}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

