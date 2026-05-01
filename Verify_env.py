"""
PneumoScan - Environment Verification Script
Run this after installing requirements.txt to confirm everything works.
Usage: python verify_env.py
"""

import sys

results = []
failed = []

def check(library_name, import_name, version_attr="__version__"):
    try:
        mod = __import__(import_name)
        version = getattr(mod, version_attr, "ok")
        results.append((library_name, str(version), "PASS"))
    except ImportError as e:
        results.append((library_name, str(e), "FAIL"))
        failed.append(library_name)

# Core checks
check("Python",         "sys",              "version")
check("torch",          "torch")
check("torchvision",    "torchvision")
check("timm",           "timm")
check("PySide6",        "PySide6")
check("opencv",         "cv2",              "__version__")
check("albumentations", "albumentations")
check("Pillow",         "PIL",              "__version__")
check("pymongo",        "pymongo")
check("scikit-learn",   "sklearn",          "__version__")
check("pandas",         "pandas")
check("numpy",          "numpy")
check("matplotlib",     "matplotlib")
check("tqdm",           "tqdm")
check("reportlab",      "reportlab",        "Version")
check("bcrypt",         "bcrypt")
check("grad-cam",       "pytorch_grad_cam")
check("PyYAML",         "yaml",             "__version__")

# Print results table
print()
print("=" * 55)
print("  PneumoScan — Environment Verification")
print("=" * 55)
print(f"  {'Library':<20} {'Version':<20} {'Status'}")
print("-" * 55)
for name, version, status in results:
    version_display = version[:35] if len(version) > 35 else version
    marker = "OK" if status == "PASS" else "FAIL"
    print(f"  {name:<20} {version_display:<20} {marker}")
print("=" * 55)

# GPU check
try:
    import torch
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU: Not available — CPU mode (expected for this project)")
except:
    pass

# Final verdict
print()
if not failed:
    print("  ALL CHECKS PASSED. Your environment is ready.")
    print("  You can now proceed to Step 2: Dataset Preparation.")
else:
    print(f"  FAILED: {', '.join(failed)}")
    print("  Run: pip install -r requirements.txt  and try again.")
print()