# make_small_dataset.py
# Copies a small subset (20–50 classes) from the full dataset into a new folder "sequences_small"

import os
import shutil
from pathlib import Path

# --- CONFIG ---
SOURCE_BASE = Path(r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart")
TARGET_BASE = Path(r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_small")
NUM_CLASSES = 30   # adjust to 20, 30, or 50 as desired

# --- MAIN ---
splits = ["train", "val", "test"]

for split in splits:
    src_split = SOURCE_BASE / split
    dst_split = TARGET_BASE / split
    dst_split.mkdir(parents=True, exist_ok=True)

    # list classes alphabetically
    classes = sorted([d for d in os.listdir(src_split) if (src_split / d).is_dir()])
    classes = classes[:NUM_CLASSES]
    print(f"[{split}] Copying {len(classes)} classes...")

    for cls in classes:
        src_cls = src_split / cls
        dst_cls = dst_split / cls
        if dst_cls.exists():
            print(f" - Skipping existing {cls}")
            continue
        shutil.copytree(src_cls, dst_cls)
        print(f" - Copied {cls}")

print("\n✅ Done. Small dataset created at:", TARGET_BASE)
