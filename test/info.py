# python
import sys
import platform
import os
import torch

def get_pkg_version(pkg_name, import_name=None):
    try:
        # 优先使用 importlib.metadata (Python 3.8+)
        try:
            from importlib import metadata
        except Exception:
            import importlib_metadata as metadata  # type: ignore
        return metadata.version(pkg_name)
    except Exception:
        try:
            import pkg_resources
            return pkg_resources.get_distribution(pkg_name).version
        except Exception:
            try:
                m = __import__(import_name or pkg_name)
                return getattr(m, "__version__", "unknown")
            except Exception:
                return "not installed"

def main():
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Machine:", platform.machine(), "Processor:", platform.processor())
    print("CPU count:", os.cpu_count())

    # PyTorch / CUDA
    print("torch:", getattr(torch, "__version__", "not installed"))
    print("torchvision:", get_pkg_version("torchvision", "torchvision"))
    print("CUDA available:", torch.cuda.is_available())
    print("torch.cuda.version:", torch.version.cuda)
    try:
        print("cuDNN version:", torch.backends.cudnn.version())
    except Exception:
        print("cuDNN version: unknown")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            except Exception:
                print(f"GPU {i}: name unknown")

    # 常用包版本
    pkgs = [("numpy", "numpy"), ("Pillow", "PIL"), ("opencv-python", "cv2")]
    for pkg, imp in pkgs:
        print(f"{pkg}:", get_pkg_version(pkg, imp))

if __name__ == "__main__":
    main()
