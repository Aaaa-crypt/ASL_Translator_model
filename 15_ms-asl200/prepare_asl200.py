# prepare_msasl200.py
import os
import json
import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\Users\garga\Documents\Maturarbeit\MS-ASL200"  # Where frames will be saved
JSON_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\MS-ASL"  # Where MSASL_*.json files are
FRAMES_PER_CLIP = 12
SUBSET_CLASSES = 200  # MS-ASL200
APPLY_BBOX = True  # set False to skip bounding box crop

# ---------------- HELPERS ----------------
def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # filter for subset classes
    return [d for d in data if d['label'] < SUBSET_CLASSES]

def download_clip(url, start_time, end_time, out_path):
    """
    Download a video clip segment using yt-dlp and ffmpeg
    """
    os.makedirs(out_path.parent, exist_ok=True)
    tmp_video = out_path.parent / f"tmp_{out_path.stem}.mp4"
    
    if not tmp_video.exists():
        cmd_dl = ["yt-dlp", "-f", "mp4", "-o", str(tmp_video), url]
        subprocess.run(cmd_dl, check=True)
    
    if not out_path.exists():
        cmd_ffmpeg = [
            "ffmpeg", "-y",
            "-i", str(tmp_video),
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264", "-c:a", "aac",
            str(out_path)
        ]
        subprocess.run(cmd_ffmpeg, check=True)
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

def process_subset(json_file, split_name):
    data = load_json(json_file)
    print(f"[INFO] Processing {split_name}: {len(data)} clips")
    for entry in tqdm(data):
        label = entry['label']
        url = entry['url']
        start_time = entry['start_time']
        end_time = entry['end_time']
        bbox = entry.get('box')  # normalized bbox [y0, x0, y1, x1]
        
        out_dir = Path(DATA_DIR) / split_name / str(label) / entry.get('url').split('=')[-1]
        video_file = out_dir / "clip.mp4"
        frames_dir = out_dir / "frames"
        
        try:
            download_clip(url, start_time, end_time, video_file)
            extract_frames(video_file, frames_dir, num_frames=FRAMES_PER_CLIP, bbox=bbox)
        except Exception as e:
            print(f"[WARN] Failed clip {url}: {e}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    process_subset(os.path.join(JSON_DIR, "MSASL_train.json"), "train")
    process_subset(os.path.join(JSON_DIR, "MSASL_val.json"), "val")
    process_subset(os.path.join(JSON_DIR, "MSASL_test.json"), "test")
    
    print("[INFO] Dataset preparation complete!")
