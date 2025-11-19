# debug_train_ds_shapes.py
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
BATCH_SIZE = 16         # batch size for TF dataset

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
    return tf.stack(clip)


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
          f"min={clip.numpy().min():.3f}, max={clip.numpy().max():.3f}, mean={clip.numpy().mean():.3f}")

# ------------------------------------------------------
# CREATE TF DATASET & CHECK ONE BATCH
# ------------------------------------------------------
def tf_load_clip_py(frame_paths, label):
    clip = load_clip_preview(frame_paths)
    label_onehot = tf.one_hot(label, depth=len(train_classes))
    return clip, label_onehot

# Create TF dataset
train_ds = tf.data.Dataset.from_generator(
    lambda: ((frames, lbl) for frames, lbl, _ in train_clips),
    output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
train_ds = train_ds.map(
    lambda frames, lbl: tf.py_function(tf_load_clip_py, [frames, lbl], [tf.float32, tf.float32])
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Debug: check one batch
images, labels = next(iter(train_ds))
print("\n=== DEBUG: ONE BATCH SHAPE ===")
print("images.shape:", images.shape)
print("labels.shape:", labels.shape)

print("\n✅ Debugging complete.")
