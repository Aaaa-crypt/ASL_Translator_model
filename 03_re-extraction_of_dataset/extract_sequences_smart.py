# ------------------------ extract_sequences_smart.py ------------------------
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# -------- CONFIG --------
DATASET_ROOT = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen"
SPLITS_DIR   = os.path.join(DATASET_ROOT, "splits")
VIDEOS_DIR   = os.path.join(DATASET_ROOT, "videos")
OUTPUT_ROOT  = os.path.join(DATASET_ROOT, "sequences_smart")  # where sequences will be written

FRAMES_PER_CLIP = 6          # number of frames per video
IMG_SIZE = (112, 112)        # resize frames

VIDEO_COL = "Video file"     # CSV column with filenames
LABEL_COL = "Gloss"          # CSV column with labels

# -------- helpers --------
def smart_indices(num_frames, k):
    """
    Select k frames from the middle portion of the video using a "------I---I--I--I--I---I------" pattern.
    """
    if num_frames <= 0:
        return []

    # Ignore first 10% and last 10%
    start = int(num_frames * 0.1)
    end = int(num_frames * 0.9)
    middle_frames = max(end - start, 1)

    if middle_frames >= k:
        indices = np.linspace(start, end - 1, k, dtype=int)
    else:
        # Repeat last frame if not enough
        indices = list(range(start, end)) + [end - 1] * (k - middle_frames)
    return indices.tolist()

def sanitize_stem(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    return "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_"))

def save_frames_from_video(video_path, out_dir, indices, img_size):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "cannot open video"

    saved = 0
    W, H = img_size
    indices_set = set(indices)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices_set:
            frame_resized = cv2.resize(frame, (W, H))
            out_path = os.path.join(out_dir, f"frame_{saved+1:04d}.jpg")
            cv2.imwrite(out_path, frame_resized)
            saved += 1
        frame_idx += 1

    cap.release()
    if saved < len(indices):
        return False, f"only {saved}/{len(indices)} frames saved"
    return True, "ok"

def process_split(split_name):
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    if not os.path.isfile(csv_path):
        print(f"[WARN] CSV missing: {csv_path}")
        return

    print(f"[INFO] Processing {split_name}...")
    df = pd.read_csv(csv_path)
    errors = 0
    processed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract {split_name}"):
        video_file = str(row[VIDEO_COL])
        label = str(row[LABEL_COL]).strip()
        video_path = os.path.join(VIDEOS_DIR, os.path.basename(video_file))

        if not os.path.isfile(video_path):
            errors += 1
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            errors += 1
            continue

        indices = smart_indices(total_frames, FRAMES_PER_CLIP)
        seq_name = f"{sanitize_stem(video_file)}_clip01"
        out_dir = os.path.join(OUTPUT_ROOT, split_name, label, seq_name)

        ok, msg = save_frames_from_video(video_path, out_dir, indices, IMG_SIZE)
        if not ok:
            errors += 1
        processed += 1

    print(f"[DONE] {split_name}: processed={processed}, errors={errors}")

# -------- main --------
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for split in ["train", "val", "test"]:
        process_split(split)
    print("[ALL DONE] Sequences saved.")
