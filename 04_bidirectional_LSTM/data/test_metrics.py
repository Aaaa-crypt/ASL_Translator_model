# test_metrics.py

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from PIL import Image

# ---------------- USER CONFIG ----------------
MODEL_PATH = r"C:\Users\garga\Documents\Maturarbeit\04_bidirectional_LSTM\asl_seq_model_smart_v3.keras"
TEST_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart\test"
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\04_bidirectional_LSTM"

TEST_JSON_PATH = os.path.join(OUTPUT_DIR, "test_metrics.json")
MAX_CLASSES = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
print(f"[INFO] Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Automatically detect model input shape
# Model input shape format: (None, FRAMES, H, W, C)
FRAMES_PER_CLIP = model.input_shape[1]
IMG_HEIGHT = model.input_shape[2]
IMG_WIDTH = model.input_shape[3]
CHANNELS = model.input_shape[4]
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)  # PIL expects (width, height)

print(f"[INFO] Model expects {FRAMES_PER_CLIP} frames of size {IMG_WIDTH}x{IMG_HEIGHT} with {CHANNELS} channels.")

# ---------------- HELPERS ----------------
def load_clip(clip_dir):
    frames = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith(".jpg")])
    clip = []

    if len(frames) == 0:
        print(f"[WARNING] No frames found in {clip_dir}. Using dummy frames.")
        dummy = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
        return np.stack([dummy] * FRAMES_PER_CLIP)

    # Load up to FRAMES_PER_CLIP frames
    for f in frames[:FRAMES_PER_CLIP]:
        img = Image.open(os.path.join(clip_dir, f)).convert("RGB").resize(IMG_SIZE)
        clip.append(np.array(img) / 255.0)

    # Pad missing frames with the last available frame
    while len(clip) < FRAMES_PER_CLIP:
        clip.append(clip[-1])

    return np.stack(clip, axis=0)


def load_test_data(test_dir):
    X, y = [], []
    
    # Restrict classes (optional)
    class_names = sorted(
        [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    )[:MAX_CLASSES]

    # Map only these classes
    class_map = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        for clip_name in os.listdir(cls_dir):
            clip_path = os.path.join(cls_dir, clip_name)
            if os.path.isdir(clip_path):
                clip = load_clip(clip_path)
                X.append(clip)
                y.append(class_map[cls])

    return np.array(X), np.array(y), class_names


# ---------------- LOAD TEST DATA ----------------
print("[INFO] Loading test data...")
X_test, y_test, class_names = load_test_data(TEST_DIR)

NUM_CLASSES = model.output_shape[-1]  # ensures match with trained model

print(f"[INFO] Loaded {len(X_test)} test clips.")
print(f"[INFO] Test set classes: {len(class_names)}, Model output classes: {NUM_CLASSES}")

if len(class_names) != NUM_CLASSES:
    print("[WARNING] Test classes do NOT match model classes. Padding one-hot labels.")

y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

# ---------------- EVALUATE MODEL ----------------
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"[INFO] Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

# ---------------- SAVE TEST METRICS ----------------
metrics = {
    "loss": float(loss),
    "accuracy": float(acc),
    "num_test_samples": int(len(X_test)),
    "test_classes_found": int(len(class_names)),
    "model_output_classes": int(NUM_CLASSES),
    "frames_per_clip": int(FRAMES_PER_CLIP),
    "image_width": int(IMG_WIDTH),
    "image_height": int(IMG_HEIGHT)
}

with open(TEST_JSON_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Test metrics saved to: {TEST_JSON_PATH}")
