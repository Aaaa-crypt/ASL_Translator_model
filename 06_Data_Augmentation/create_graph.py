# ------------------------ plot_test_bar_from_model.py ------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from PIL import Image

# -------- USER CONFIG --------
MODEL_PATH = r"C:\Users\garga\Documents\Maturarbeit\05_re-extraction_dataset_again\asl_seq_model_smart_v3.keras"
TEST_DATA_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart\test"  # Folder with test clips organized in class subfolders
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEST_BAR_PLOT_PATH = os.path.join(OUTPUT_DIR, "test_bar_graph_v3.png")
TEST_BAR_COLOR = "#1f77b4"

# ---------------- HELPER FUNCTIONS ----------------
def load_clip(clip_dir):
    """Load a single clip as a numpy array of shape (FRAMES_PER_CLIP, H, W, 3)."""
    frames = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith(".jpg")])[:FRAMES_PER_CLIP]
    if len(frames) < FRAMES_PER_CLIP:
        return None
    clip = []
    for f in frames:
        path = os.path.join(clip_dir, f)
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        clip.append(np.array(img) / 255.0)
    return np.stack(clip)

def load_test_dataset(test_dir):
    """Load all test clips and labels."""
    clips = []
    labels = []
    class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    class_map = {c: i for i, c in enumerate(class_names)}
    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        for clip_dir in sorted(os.listdir(cls_dir)):
            full_clip_dir = os.path.join(cls_dir, clip_dir)
            if os.path.isdir(full_clip_dir):
                clip = load_clip(full_clip_dir)
                if clip is not None:
                    clips.append(clip)
                    labels.append(class_map[cls])
    return np.array(clips), to_categorical(labels, num_classes=len(class_names))

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# ---------------- LOAD TEST DATA ----------------
X_test, y_test = load_test_dataset(TEST_DATA_DIR)
print(f"[INFO] Loaded {len(X_test)} test clips.")

# ---------------- EVALUATE ----------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"[INFO] Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# ---------------- PLOT TEST BAR GRAPH ----------------
plt.figure(figsize=(6,4))
plt.bar(["Accuracy","Loss"], [test_acc, test_loss], color=TEST_BAR_COLOR)
plt.title("Test Metrics")
plt.ylabel("Value")
plt.ylim(0, max(test_acc, test_loss)*1.2)
plt.savefig(TEST_BAR_PLOT_PATH)
plt.show()
print(f"✅ Test bar graph saved to {TEST_BAR_PLOT_PATH}")
