import os
import shutil

# -------------------- PATHS --------------------
SMALL_BASE = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\asl_sequences\kaggle\working\sequences"
BIG_BASE   = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
OUTPUT_BASE = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\asl_sequences_mixed"

splits = ["train", "val", "test"]

# Create base output folders
for split in splits:
    os.makedirs(os.path.join(OUTPUT_BASE, split), exist_ok=True)

# -------------------- HELPERS --------------------
def copy_class_folder(src_class_path, dst_class_path):
    """
    Copies clip folders from src_class_path → dst_class_path.
    If a clip folder exists already, it auto-renames it.
    """
    if not os.path.exists(src_class_path):
        return  # nothing to copy

    os.makedirs(dst_class_path, exist_ok=True)

    for clip in os.listdir(src_class_path):
        src_clip = os.path.join(src_class_path, clip)
        if not os.path.isdir(src_clip):
            continue

        dst_clip = os.path.join(dst_class_path, clip)

        # Avoid overwrite (rename)
        if os.path.exists(dst_clip):
            base = clip
            i = 1
            while os.path.exists(dst_clip):
                dst_clip = os.path.join(dst_class_path, f"{base}_mix{i}")
                i += 1

        shutil.copytree(src_clip, dst_clip)
        print(f"[COPY] {src_clip} --> {dst_clip}")


# -------------------- MAIN --------------------
print("\n[INFO] Reading SMALL dataset class names from train split...")
small_train_dir = os.path.join(SMALL_BASE, "train")
small_classes = sorted(
    [d for d in os.listdir(small_train_dir)
     if os.path.isdir(os.path.join(small_train_dir, d))]
)

print(f"[INFO] Found {len(small_classes)} classes in SMALL dataset.")

for cls in small_classes:
    print("\n==============================")
    print(f"[CLASS] {cls}")
    print("==============================")

    for split in splits:
        small_split_cls = os.path.join(SMALL_BASE, split, cls)
        big_split_cls   = os.path.join(BIG_BASE, split, cls)
        out_split_cls   = os.path.join(OUTPUT_BASE, split, cls)

        print(f"\n[PROCESS] Split: {split}")

        # 1. Copy small dataset clips
        if os.path.exists(small_split_cls):
            print(f"[MIX] Copying SMALL dataset → {split}")
            copy_class_folder(small_split_cls, out_split_cls)
        else:
            print(f"[WARN] SMALL dataset has no {split} for {cls}")

        # 2. Copy big dataset clips
        if os.path.exists(big_split_cls):
            print(f"[MIX] Copying BIG dataset → {split}")
            copy_class_folder(big_split_cls, out_split_cls)
        else:
            print(f"[WARN] BIG dataset has no matching {split} for {cls}")

print("\n🎉 DONE! Full mixed dataset with train/val/test created.")
print(f"Output at: {OUTPUT_BASE}")
