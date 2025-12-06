
# ------------------------ test_metrics.py ------------------------
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt

# -------- CONFIG (MATCH TRAINING) --------
SEQUENCES_DIR = r"C:\Users\GG\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
TEST_DIR = os.path.join(SEQUENCES_DIR, "test")

MODEL_PATH = r"C:\Users\GG\Documents\Maturarbeit\2\result_of_seq1\asl_seq_model.keras"
OUTPUT_DIR = r"C:\Users\GG\Documents\Maturarbeit\2\results "

TEST_JSON_PATH = os.path.join(OUTPUT_DIR, "test_metrics.json")
BAR_PLOT_PATH = os.path.join(OUTPUT_DIR, "test_bar_graph.png")

FRAMES_PER_CLIP = 5
IMG_SIZE = (112, 112)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
print(f"[INFO] Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

NUM_CLASSES = model.output_shape[-1]

# ---------------- HELPERS ----------------
def load_clip(clip_dir):
    frame_files = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith(".jpg")])
    frames = []

    if len(frame_files) == 0:
        dummy = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        return np.stack([dummy] * FRAMES_PER_CLIP).astype(np.float32)

    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_files) - 1)
        img_path = os.path.join(clip_dir, frame_files[idx])
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        frames.append((np.array(img) / 255.0).astype(np.float32))

    return np.stack(frames, axis=0).astype(np.float32)

def load_test_data(test_dir):
    X, y = [], []

    # Removed MAX_CLASSES entirely
    class_names = sorted(
        [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    )
    class_map = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        for clip_folder in os.listdir(cls_dir):
            clip_path = os.path.join(cls_dir, clip_folder)
            if os.path.isdir(clip_path):
                X.append(load_clip(clip_path))
                y.append(class_map[cls])

    X = np.array(X, dtype=np.float32)   # <-- FIXED: no float64
    y = np.array(y, dtype=np.int32)

    return X, y, class_names

# ---------------- LOAD TEST DATA ----------------
print("[INFO] Loading test data...")
X_test, y_test, class_names = load_test_data(TEST_DIR)

print(f"[INFO] Loaded {len(X_test)} test clips.")
print(f"[INFO] Model outputs {NUM_CLASSES} classes; test set has {len(class_names)} classes.")

y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

# ---------------- EVALUATE ----------------
loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"[INFO] Test Accuracy: {acc:.4f}")
print(f"[INFO] Test Loss: {loss:.4f}")

# ---------------- SAVE METRICS JSON ----------------
metrics = {
    "loss": float(loss),
    "accuracy": float(acc),
    "num_test_samples": int(len(X_test)),
    "num_classes": int(NUM_CLASSES)
}

with open(TEST_JSON_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Saved test metrics → {TEST_JSON_PATH}")

# ---------------- MAKE BAR GRAPH ----------------
plt.figure(figsize=(6, 4))
plt.bar(["Accuracy", "Loss"], [acc, loss])
plt.title("Model Test Performance")
plt.ylabel("Value")
plt.ylim(0, max(acc, loss) * 1.2)

for i, value in enumerate([acc, loss]):
    plt.text(i, value + 0.01, f"{value:.3f}", ha='center')

plt.tight_layout()
plt.savefig(BAR_PLOT_PATH)
plt.close()

print(f"📊 Saved test bar graph → {BAR_PLOT_PATH}")
print("🎉 Done!")
