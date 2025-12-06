# test_metrics.py

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
MODEL_PATH = r"C:\Users\garga\Documents\Maturarbeit\5\asl_seq_model_smart_v3.keras"
TEST_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart\test"
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\5"

TEST_JSON_PATH = os.path.join(OUTPUT_DIR, "test_metrics.json")
BAR_PLOT_PATH = os.path.join(OUTPUT_DIR, "test_bar_graph.png")

MAX_CLASSES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
print(f"[INFO] Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# 🔥 FIX: compile the model so evaluate() works
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

FRAMES_PER_CLIP = model.input_shape[1]
IMG_HEIGHT = model.input_shape[2]
IMG_WIDTH = model.input_shape[3]
CHANNELS = model.input_shape[4]
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# ---------------- HELPERS ----------------
def load_clip(clip_dir):
    frames = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith(".jpg")])
    clip = []

    if len(frames) == 0:
        dummy = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
        return np.stack([dummy] * FRAMES_PER_CLIP)

    for f in frames[:FRAMES_PER_CLIP]:
        img = Image.open(os.path.join(clip_dir, f)).convert("RGB").resize(IMG_SIZE)
        clip.append(np.array(img) / 255.0)

    while len(clip) < FRAMES_PER_CLIP:
        clip.append(clip[-1])

    return np.stack(clip)

def load_test_data(test_dir):
    X, y = [], []

    class_names = sorted(
        [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    )[:MAX_CLASSES]

    class_map = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        for clip_folder in os.listdir(cls_dir):
            clip_path = os.path.join(cls_dir, clip_folder)
            if os.path.isdir(clip_path):
                X.append(load_clip(clip_path))
                y.append(class_map[cls])

    return np.array(X), np.array(y), class_names

# ---------------- LOAD TEST DATA ----------------
print("[INFO] Loading test data...")
X_test, y_test, class_names = load_test_data(TEST_DIR)

NUM_CLASSES = model.output_shape[-1]
y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

print(f"[INFO] Loaded {len(X_test)} test clips.")

# ---------------- EVALUATE MODEL ----------------
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"[INFO] Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

# ---------------- SAVE SIMPLE METRICS JSON ----------------
metrics = {
    "loss": float(loss),
    "accuracy": float(acc),
    "num_test_samples": int(len(X_test)),
    "num_classes": int(NUM_CLASSES)
}

with open(TEST_JSON_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Saved metrics JSON → {TEST_JSON_PATH}")

# ---------------- CREATE BAR GRAPH ----------------
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy", "Loss"], [acc, loss])
plt.title("Model Test Performance")
plt.ylabel("Value")
plt.ylim(0, max(acc, loss) * 1.2)

for i, value in enumerate([acc, loss]):
    plt.text(i, value + 0.01, f"{value:.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(BAR_PLOT_PATH)
plt.close()

print(f"📊 Saved bar graph → {BAR_PLOT_PATH}")
print("🎉 Done!")
