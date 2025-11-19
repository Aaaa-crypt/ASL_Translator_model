import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import Counter

# ---------------- CONFIG ----------------
BASE = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
IMG_SIZE = (96, 96)
FRAMES_PER_CLIP = 12
NUM_SAMPLES = 5         # how many clips to fully load for inspection
MAX_CLASSES = 200       # limit classes for debugging

# ------------------------------------------------------
# HELPERS
# ------------------------------------------------------
def get_clip_paths(split, max_classes=None):
    split_dir = os.path.join(BASE, split)
    class_names = sorted([d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d))])

    if max_classes:
        class_names = class_names[:max_classes]

    class_map = {c: i for i, c in enumerate(class_names)}
    clip_paths = []

    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir)
                       if os.path.isdir(os.path.join(cls_dir, d))])

        for vid in vids:
            vid_dir = os.path.join(cls_dir, vid)
            frames = sorted(
                os.path.join(vid_dir, f)
                for f in os.listdir(vid_dir)
                if f.lower().endswith(".jpg")
            )
            if frames:
                clip_paths.append((frames, class_map[cls], cls))

    return clip_paths, class_names, class_map


def load_clip_preview(frame_paths):
    """Load & preprocess frames for detailed inspection."""
    clip = []
    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_paths) - 1)
        img_raw = tf.io.read_file(frame_paths[idx])
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        clip.append(img)

    return tf.stack(clip).numpy()


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
train_clips, train_classes, class_map = get_clip_paths("train", max_classes=MAX_CLASSES)
val_clips, val_classes, _ = get_clip_paths("train", max_classes=MAX_CLASSES)  # Using TRAIN as VAL intentionally

print(f"[INFO] Loaded: {len(train_clips)} train clips")
print(f"[INFO] Using first {len(train_classes)} classes")

# ------------------------------------------------------
# CLASS NAME CONSISTENCY
# ------------------------------------------------------
print("\n=== CHECK: TRAIN vs VAL CLASS NAMES ===")

if train_classes == val_classes:
    print("✔ Train and Val classes match exactly.")
else:
    print("❌ Class mismatch!")
    missing_train = set(val_classes) - set(train_classes)
    missing_val = set(train_classes) - set(val_classes)
    print(" Only in VAL:", missing_val)
    print(" Only in TRAIN:", missing_train)

# ------------------------------------------------------
# CLASS → INDEX MAP
# ------------------------------------------------------
print("\n=== CLASS → INDEX MAP (first 30) ===")
for i, (name, idx) in enumerate(class_map.items()):
    if i >= 30:
        print("... (truncated)")
        break
    print(f"{idx:3d}: {name}")

# ------------------------------------------------------
# ONE EXAMPLE PER CLASS (NO IMAGE LOAD)
# ------------------------------------------------------
print("\n=== ONE EXAMPLE PER CLASS (fast check) ===")
seen = set()
for frames, idx, cls in train_clips:
    if idx not in seen:
        print(f"Class {idx:3d} ({cls}): first frame = {frames[0]}")
        seen.add(idx)
    if len(seen) == len(train_classes):
        break

# ------------------------------------------------------
# CLASS DISTRIBUTION SUMMARY
# ------------------------------------------------------
print("\n=== CLASS DISTRIBUTION (TRAIN) ===")
train_label_ids = [lbl for _, lbl, _ in train_clips]
counts = Counter(train_label_ids)

for cls_idx, count in list(counts.items())[:20]:
    print(f"Class {cls_idx:3d}: {count:3d} clips")
if len(counts) > 20:
    print("... (truncated)")

# ------------------------------------------------------
# FULL DEBUG: LOAD SAMPLE CLIPS
# ------------------------------------------------------
print("\n=== FULL DEBUG: LOADING SAMPLE TRAIN CLIPS ===")
for frames, label, cls in train_clips[:NUM_SAMPLES]:
    clip = load_clip_preview(frames)
    print(f"[{cls}] label={label}, shape={clip.shape}, "
          f"min={clip.min():.3f}, max={clip.max():.3f}, mean={clip.mean():.3f}")

print("\n=== FULL DEBUG: LOADING SAMPLE VAL CLIPS ===")
for frames, label, cls in val_clips[:NUM_SAMPLES]:
    clip = load_clip_preview(frames)
    print(f"[{cls}] label={label}, shape={clip.shape}, "
          f"min={clip.min():.3f}, max={clip.max():.3f}, mean={clip.mean():.3f}")

print("\n✅ Debugging complete.")
