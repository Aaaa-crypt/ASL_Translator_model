# create_graph.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
MODEL_PATH = r"C:\Users\garga\Documents\Maturarbeit\04_bidirectional_LSTM\asl_seq_model_smart_v3.keras"
TEST_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart\test"
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\04_bidirectional_LSTM"
BAR_PLOT_PATH = os.path.join(OUTPUT_DIR, "test_bar_graph_v3.png")
MAX_CLASSES = 100
TEST_BAR_COLOR = "#1f77b4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
print(f"[INFO] Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Automatically detect model input shape
FRAMES_PER_CLIP = model.input_shape[1]
IMG_HEIGHT = model.input_shape[2]
IMG_WIDTH = model.input_shape[3]
CHANNELS = model.input_shape[4]
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

print(f"[INFO] Model expects {FRAMES_PER_CLIP} frames per clip of size {IMG_WIDTH}x{IMG_HEIGHT} with {CHANNELS} channels.")

# ---------------- HELPERS ----------------
def load_clip(clip_dir):
    frames = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith(".jpg")])
    clip = []
    
    # Load up to FRAMES_PER_CLIP frames
    for f in frames[:FRAMES_PER_CLIP]:
        img = Image.open(os.path.join(clip_dir, f)).convert("RGB").resize(IMG_SIZE)
        clip.append(np.array(img)/255.0)
    
    # Pad with last frame if fewer frames than required
    while len(clip) < FRAMES_PER_CLIP:
        clip.append(clip[-1])
    
    return np.stack(clip)

def load_test_data(test_dir):
    X, y = [], []
    class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])[:MAX_CLASSES]
    class_map = {c:i for i,c in enumerate(class_names)}
    
    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        for clip_name in os.listdir(cls_dir):
            clip_path = os.path.join(cls_dir, clip_name)
            if os.path.isdir(clip_path):
                clip = load_clip(clip_path)
                if clip is not None:
                    X.append(clip)
                    y.append(class_map[cls])
    
    return np.array(X), np.array(y), class_names

# ---------------- LOAD TEST DATA ----------------
print("[INFO] Loading test data...")
X_test, y_test, class_names = load_test_data(TEST_DIR)
y_test_cat = to_categorical(y_test, num_classes=len(class_names))
print(f"[INFO] Loaded {len(X_test)} test clips across {len(class_names)} classes.")

# ---------------- EVALUATE MODEL ----------------
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"[INFO] Test Accuracy: {acc:.3f}, Test Loss: {loss:.3f}")

# ---------------- PLOT TEST BAR GRAPH ----------------
plt.figure(figsize=(6,4))
plt.bar(["Accuracy","Loss"], [acc, loss], color=TEST_BAR_COLOR)
plt.title("Test Metrics (Model v3)")
plt.ylabel("Value")
plt.ylim(0, max(acc, loss) * 1.2)

# Add value labels on bars
for i, v in enumerate([acc, loss]):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(BAR_PLOT_PATH)
plt.show()
print(f"✅ Test bar graph saved to: {BAR_PLOT_PATH}")
