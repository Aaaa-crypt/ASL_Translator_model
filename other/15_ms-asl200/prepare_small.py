# prepare_small.py
import os
import json
import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm
import shutil

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\Users\garga\Documents\Maturarbeit\MS-ASL200"
JSON_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\MS-ASL"
FRAMES_PER_CLIP = 12
SUBSET_CLASSES = 200
APPLY_BBOX = True

# Paths to FFmpeg binaries
FFMPEG_PATH = r"C:\Users\garga\Documents\Maturarbeit\ffmpeg-2025-11-10-git-133a0bcb13-full_build\ffmpeg-2025-11-10-git-133a0bcb13-full_build\bin\ffmpeg.exe"

# yt-dlp binary
YT_DLP = "yt-dlp"

# Check yt-dlp
if not shutil.which(YT_DLP):
    raise EnvironmentError(
        "❌ yt-dlp not found! Install it with 'pip install yt-dlp' and ensure it's in PATH."
    )

# ---------------- HELPERS ----------------
def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [d for d in data if d['label'] < SUBSET_CLASSES]

def download_clip(url, start_time, end_time, out_path):
    os.makedirs(out_path.parent, exist_ok=True)
    tmp_video = out_path.parent / f"tmp_{out_path.stem}.mp4"

    if not tmp_video.exists():
        cmd_dl = [YT_DLP, "-f", "mp4", "-o", str(tmp_video), url]
        subprocess.run(cmd_dl, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not out_path.exists():
        cmd_ffmpeg = [
            FFMPEG_PATH, "-y",
            "-i", str(tmp_video),
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264", "-c:a", "aac",
            str(out_path)
        ]
        subprocess.run(cmd_ffmpeg, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_path

def extract_frames(video_path, frames_dir, num_frames=12, bbox=None):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    idx_set = set(frame_indices)
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in idx_set:
            if bbox and APPLY_BBOX:
                h, w = frame.shape[:2]
                y0, x0, y1, x1 = bbox
                x0, x1 = int(x0 * w), int(x1 * w)
                y0, y1 = int(y0 * h), int(y1 * h)
                frame = frame[y0:y1, x0:x1]
            frame_file = frames_dir / f"{saved_count:02d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            saved_count += 1
        frame_count += 1
    cap.release()

def process_subset(json_file, split_name, max_clips=2):
    data = load_json(json_file)[:max_clips]
    print(f"[INFO] Processing {split_name}: {len(data)} clips")
    for entry in tqdm(data, desc=f"{split_name} clips", unit="clip", ncols=100):
        label = entry['label']
        url = entry['url']
        start_time = entry['start_time']
        end_time = entry['end_time']
        bbox = entry.get('box')

        out_dir = Path(DATA_DIR) / split_name / str(label) / entry.get('url').split('=')[-1]
        video_file = out_dir / "clip.mp4"
        frames_dir = out_dir / "frames"

        try:
            download_clip(url, start_time, end_time, video_file)
            extract_frames(video_file, frames_dir, num_frames=FRAMES_PER_CLIP, bbox=bbox)
            print(f"[SUCCESS] {url}")
        except Exception:
            print(f"[FAILED] {url}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    process_subset(os.path.join(JSON_DIR, "MSASL_train.json"), "train", max_clips=2)
    process_subset(os.path.join(JSON_DIR, "MSASL_val.json"), "val", max_clips=2)
    process_subset(os.path.join(JSON_DIR, "MSASL_test.json"), "test", max_clips=2)

    print("[INFO] Small test dataset preparation complete!")
