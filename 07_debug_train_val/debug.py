import os
import random
import shutil

# ---------------- CONFIG ----------------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
DEBUG_VAL_DIR = os.path.join(SEQUENCES_DIR, "debug_val2")
MAX_CLASSES = 200
TRAIN_RATIO = 0.8  # fraction of debug_val clips sampled from train
FRAMES_PER_CLIP = 12  # just for reference

# ---------------- UTILS ----------------
def get_class_dirs(base_dir):
    class_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if MAX_CLASSES:
        class_dirs = class_dirs[:MAX_CLASSES]
    return class_dirs

def collect_clips(base_dir, classes):
    clips = []
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        for vid_dir in os.listdir(cls_dir):
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            if os.path.isdir(full_vid_dir):
                frames = [f for f in os.listdir(full_vid_dir) if f.lower().endswith(".jpg")]
                if frames:
                    clips.append((cls, vid_dir, full_vid_dir))
    return clips

def copy_clip(cls, vid_dir, src_dir, dest_base):
    dest_cls_dir = os.path.join(dest_base, cls)
    os.makedirs(dest_cls_dir, exist_ok=True)
    dest_clip_dir = os.path.join(dest_cls_dir, vid_dir)
    if not os.path.exists(dest_clip_dir):
        shutil.copytree(src_dir, dest_clip_dir)
    else:
        print(f"[WARN] Clip already exists in debug_val: {dest_clip_dir}")

# ---------------- MAIN ----------------
# Clean or create debug_val folder
if os.path.exists(DEBUG_VAL_DIR):
    shutil.rmtree(DEBUG_VAL_DIR)
os.makedirs(DEBUG_VAL_DIR, exist_ok=True)

train_classes = get_class_dirs(os.path.join(SEQUENCES_DIR, "train"))
val_classes = get_class_dirs(os.path.join(SEQUENCES_DIR, "val"))

# Take only common classes (like your training setup)
common_classes = sorted(list(set(train_classes) & set(val_classes)))
print(f"[INFO] Using {len(common_classes)} common classes for debug_val")

train_clips = collect_clips(os.path.join(SEQUENCES_DIR, "train"), common_classes)
val_clips = collect_clips(os.path.join(SEQUENCES_DIR, "val"), common_classes)

# Shuffle
random.shuffle(train_clips)
random.shuffle(val_clips)

# Compute number of train clips to sample
num_val_clips = len(val_clips)
num_train_sample = int(TRAIN_RATIO * num_val_clips)

selected_train_clips = train_clips[:num_train_sample]

# Copy original val clips
for cls, vid_dir, clip_dir in val_clips:
    copy_clip(cls, vid_dir, clip_dir, DEBUG_VAL_DIR)

# Copy sampled train clips
for cls, vid_dir, clip_dir in selected_train_clips:
    copy_clip(cls, vid_dir, clip_dir, DEBUG_VAL_DIR)

print(f"[INFO] debug_val folder created at: {DEBUG_VAL_DIR}")
print(f"       {len(val_clips)} original val clips + {len(selected_train_clips)} train clips ({TRAIN_RATIO*100:.0f}% of val size)")
