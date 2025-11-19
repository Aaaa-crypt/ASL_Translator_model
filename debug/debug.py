import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIG ----------------
BASE = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
IMG_SIZE = (96, 96)
FRAMES_PER_CLIP = 12
NUM_SAMPLES = 5  # number of clips to inspect per split

# ---------------- HELPERS ----------------
def get_clip_paths(split, max_classes=None):
    split_dir = os.path.join(BASE, split)
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    if max_classes:
        class_names = class_names[:max_classes]
    class_map = {c: i for i, c in enumerate(class_names)}
    clip_paths = []
    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([os.path.join(full_vid_dir, f) for f in os.listdir(full_vid_dir) if f.lower().endswith(".jpg")])
            if frames:
                clip_paths.append((frames, class_map[cls]))
    return clip_paths, class_names

def load_clip_preview(frame_paths):
    """Load frames, resize, preprocess and return array stats"""
    clip = []
    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_paths) - 1)
        img_raw = tf.io.read_file(frame_paths[idx])
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        clip.append(img)
    clip = tf.stack(clip)
    return clip.numpy()

# ---------------- LOAD DATA ----------------
train_clips, train_classes = get_clip_paths("train", max_classes=200)
val_clips, val_classes = get_clip_paths("train", max_classes=200)  # using train as val for debugging

print(f"[INFO] Using {len(train_clips)} train clips, {len(val_clips)} val clips")

# ---------------- DEBUG SAMPLE CLIPS ----------------
print("\n=== DEBUGGING TRAIN CLIPS ===")
for frames, label in train_clips[:NUM_SAMPLES]:
    clip_arr = load_clip_preview(frames)
    print(f"Clip label: {label}, shape: {clip_arr.shape}, min: {clip_arr.min():.3f}, max: {clip_arr.max():.3f}, mean: {clip_arr.mean():.3f}")

print("\n=== DEBUGGING VAL CLIPS ===")
for frames, label in val_clips[:NUM_SAMPLES]:
    clip_arr = load_clip_preview(frames)
    print(f"Clip label: {label}, shape: {clip_arr.shape}, min: {clip_arr.min():.3f}, max: {clip_arr.max():.3f}, mean: {clip_arr.mean():.3f}")

# ---------------- SUMMARY ----------------
print("\n✅ Debugging complete. Check that train/val clip shapes and preprocessing stats match.")
