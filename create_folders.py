"""
PneumoScan - Project Folder Creator
Run once from your project root to create the full folder structure.
Usage: python create_folders.py
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent

folders = [
    # Core AI engine
    "core/models/lung",
    "core/models/cornea",
    "core/models/bone",
    "core/inference",

    # Training scripts
    "training",

    # Domain data and checkpoints
    "domains/lung/data/raw",
    "domains/lung/data/processed/train/NORMAL",
    "domains/lung/data/processed/train/PNEUMONIA",
    "domains/lung/data/processed/val/NORMAL",
    "domains/lung/data/processed/val/PNEUMONIA",
    "domains/lung/data/processed/test/NORMAL",
    "domains/lung/data/processed/test/PNEUMONIA",
    "domains/lung/checkpoints",

    # Future domains (scaffolded, empty)
    "domains/cornea/data",
    "domains/cornea/checkpoints",
    "domains/bone/data",
    "domains/bone/checkpoints",

    # GUI
    "gui/assets",

    # Utilities
    "utils",

    # Model weight files after training
    "weights/lung",
    "weights/cornea",
    "weights/bone",

    # Generated outputs
    "outputs/reports",
    "outputs/heatmaps",
]

print()
print("=" * 50)
print("  PneumoScan — Creating Project Structure")
print("=" * 50)

created = 0
for folder in folders:
    path = ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    # Add .gitkeep so empty folders are tracked by Git
    gitkeep = path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
    print(f"  OK  {folder}")
    created += 1

print("=" * 50)
print(f"  {created} folders created successfully.")
print()
print("  Project structure is ready.")
print("  Next: open this folder in VS Code.")
print()