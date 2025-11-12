import os
import tensorflow as tf

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
MAX_CLASSES = 100

# ---------------- HELPERS ----------------
def get_clip_paths(split="train", max_classes=MAX_CLASSES):
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"❌ Split directory not found: {split_dir}")

    class_names = sorted([d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d))])[:max_classes]
    clip_paths = []

    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([os.path.join(full_vid_dir, f)
                             for f in os.listdir(full_vid_dir) if f.lower().endswith(".jpg")])
            if frames:
                clip_paths.append((frames, cls))
    return clip_paths, class_names

# ---------------- DEBUG TEST ----------------
print("✅ Running diagnostic class check...")

train_clip_paths, train_class_names = get_clip_paths("train")
val_clip_paths, val_class_names = get_clip_paths("val")

print("\n[DEBUG] First 10 TRAIN classes:")
print(train_class_names[:10])

print("\n[DEBUG] First 10 VAL classes:")
print(val_class_names[:10])

print("\n[DEBUG] TRAIN class count:", len(train_class_names))
print("[DEBUG] VAL class count:", len(val_class_names))

print("\n✅ Diagnostic complete. Send me the output.")
print("❗ Do NOT train yet — we must confirm class mismatch first.")
