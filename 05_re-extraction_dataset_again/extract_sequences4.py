# extract_sequences_smart_v3_accurate_debug.py
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ---------- CONFIG ----------
BASE_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen"
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
OUTPUT_DIR = os.path.join(BASE_DIR, "sequences_smart_v3_accurate_debug")

FRAMES_PER_CLIP = 12               
RESOLUTION = (224, 224)           
CROP_START = 0.10
CROP_END = 0.90
APPLY_SHARPEN = True
SHARPEN_SIGMA = 2.0

# MediaPipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True, model_complexity=1)

# ---------- HELPERS ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sharpen_image_cv2(image, sigma=SHARPEN_SIGMA):
    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def smart_indices(total_frames, frames_per_clip=FRAMES_PER_CLIP):
    start = int(total_frames * CROP_START)
    end = max(start + 1, int(total_frames * CROP_END))
    if end - start < frames_per_clip:
        return list(np.linspace(0, total_frames - 1, frames_per_clip, dtype=int))
    indices = np.linspace(start, end - 1, frames_per_clip + 2, dtype=int)[1:-1]
    return indices.tolist()

def get_hands_face_bbox_from_frame(frame):
    h, w = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    xs, ys = [], []

    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

    if not xs or not ys:
        return None

    x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
    y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
    pad_x = int((x2 - x1) * 0.35) + 10
    pad_y = int((y2 - y1) * 0.35) + 10
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y))

def fallback_upper_body_bbox(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 3
    box_w, box_h = int(w * 0.6), int(h * 0.6)
    return (max(0, cx - box_w // 2), max(0, cy - box_h // 2),
            min(w - 1, cx + box_w // 2), min(h - 1, cy + box_h // 2))

# ---------- VIDEO PROCESSING ----------
def process_one_video(video_name, label, split):
    print(f"[INFO] Processing video: {split}/{label}/{video_name}")
    video_path = os.path.join(VIDEOS_DIR, f"{video_name}.mp4")
    split_dir = os.path.join(OUTPUT_DIR, split)
    label_dir = os.path.join(split_dir, label)
    clip_dir = os.path.join(label_dir, f"{video_name}_clip")
    ensure_dir(clip_dir)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        print(f"[WARN] No frames in video: {video_name}")
        return (0, 1)

    indices = smart_indices(total_frames)
    bbox = None
    # try multiple frames to detect bbox
    for idx in indices[len(indices)//3 : len(indices)//3 + 3]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            bbox = get_hands_face_bbox_from_frame(frame)
            if bbox is not None:
                break
    if bbox is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[len(indices)//2]))
        ret, frame = cap.read()
        if ret and frame is not None:
            bbox = get_hands_face_bbox_from_frame(frame)
    if bbox is None:
        bbox = fallback_upper_body_bbox(frame)

    # save frames
    saved_frames = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2+1, x1:x2+1]
        if APPLY_SHARPEN:
            crop = sharpen_image_cv2(crop)
        crop = cv2.resize(crop, RESOLUTION)
        cv2.imwrite(os.path.join(clip_dir, f"frame_{saved_frames+1:04d}.jpg"), crop)
        saved_frames += 1

    cap.release()
    if saved_frames == 0:
        try:
            os.rmdir(clip_dir)
        except Exception:
            pass
        return (0, 1)
    return (1, 0)

# ---------- SPLIT PROCESSING ----------
def process_split(split):
    csv_path = os.path.join(SPLITS_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing CSV: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    print(f"\n[INFO] Processing split: {split} ({len(df)} videos)")
    processed, errors = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract {split}"):
        label = str(row["Gloss"]).strip()
        video_name = str(row["Video file"]).replace(".mp4", "").strip()
        ok, err = process_one_video(video_name, label, split)
        processed += ok
        errors += err

    print(f"[DONE] Split {split}: processed={processed}, errors={errors}")

# ---------- MAIN ----------
if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        process_split(split)
    holistic.close()
    print("[ALL DONE] Sequences saved to:", OUTPUT_DIR)
