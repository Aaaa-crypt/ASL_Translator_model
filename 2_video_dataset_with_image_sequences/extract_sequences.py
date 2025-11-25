import os
import cv2
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

# -------- CONFIG --------
DATASET_ROOT = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen"
SPLITS_DIR   = os.path.join(DATASET_ROOT, "splits")
VIDEOS_DIR   = os.path.join(DATASET_ROOT, "videos")

OUTPUT_ROOT = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences"
FRACTION = 0.2                      # use ~1/5th the videos in each split (change to 1.0 to use all)
FRAMES_PER_CLIP = 16                # number of frames per sequence
IMG_SIZE = (112, 112)               # resize WxH for saved frames

# CSV column names (adjust if your CSV uses different ones)
VIDEO_COL_CANDIDATES = ["Video file"]
LABEL_COL_CANDIDATES = ["Gloss"]

# -------- HELPERS --------
def choose_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns found in CSV: {candidates}\nFound: {df.columns.tolist()}")

def uniform_indices(num_frames, k):
    if num_frames <= 0:
        return []
    if num_frames >= k:
        return np.linspace(0, num_frames - 1, k, dtype=int).tolist()
    # if too short, repeat last frame
    idx = np.linspace(0, num_frames - 1, num_frames, dtype=int).tolist()
    idx += [idx[-1]] * (k - num_frames)
    return idx[:k]

def save_sequence_from_video(video_path, out_dir, frames_per_clip=16, img_size=(112,112)):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open video"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = uniform_indices(total, frames_per_clip)
    for i, fidx in enumerate(idxs, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, f"Failed to read frame at {fidx}"
        frame = cv2.resize(frame, img_size)  # (W,H)
        out_path = os.path.join(out_dir, f"frame_{i:04d}.jpg")
        ok2 = cv2.imwrite(out_path, frame)
        if not ok2:
            return False, f"Failed to write {out_path}"
    cap.release()
    return True, "ok"

def load_split_csv(split_name):
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    vcol = choose_col(df, VIDEO_COL_CANDIDATES)
    lcol = choose_col(df, LABEL_COL_CANDIDATES)
    return df, vcol, lcol

def sanitize_stem(name):
    # remove extension + keep safe chars
    stem = os.path.splitext(os.path.basename(name))[0]
    return "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_"))

# -------- MAIN --------
def process_split(split_name, fraction=1.0):
    df, vcol, lcol = load_split_csv(split_name)

    # take a fraction (stratified-ish by groupby)
    if 0 < fraction < 1.0:
        df = df.groupby(lcol, group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=42)).reset_index(drop=True)

    out_split_dir = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(out_split_dir, exist_ok=True)

    errors = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name}"):
        video_name = str(row[vcol])
        label = str(row[lcol])

        # If CSV stores paths, reduce to filename
        video_file = os.path.basename(video_name)
        video_path = os.path.join(VIDEOS_DIR, video_file)

        if not os.path.isfile(video_path):
            errors += 1
            continue

        safe_stem = sanitize_stem(video_file)
        seq_dir = os.path.join(out_split_dir, label, f"{safe_stem}_clip01")
        ok, msg = save_sequence_from_video(video_path, seq_dir, FRAMES_PER_CLIP, IMG_SIZE)
        if not ok:
            errors += 1

    print(f"[{split_name}] done. Errors: {errors}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    process_split("train", FRACTION)
    process_split("val",   FRACTION)   # you can set to 1.0 if you want full val/test
    process_split("test",  FRACTION)
    print("All splits processed.")