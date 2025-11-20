import os
import math
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart\test"
TEST_DIR = os.path.join(SEQUENCES_DIR, "test")
MODEL_PATH = r"C:\Users\garga\Documents\Maturarbeit\8_re-extraction_dataset_again\asl_seq_model_smart_v3.keras"
TEST_BAR_PLOT = os.path.join(os.path.dirname(MODEL_PATH), "test_bar_plot.png")

FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 8
MAX_CLASSES = 100  # must match trained model output

# ---------------- HELPERS ----------------
def get_test_clips():
    classes = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    classes = classes[:MAX_CLASSES]
    class_map = {c:i for i,c in enumerate(classes)}

    clip_list = []
    for cls in classes:
        cls_dir = os.path.join(TEST_DIR, cls)
        for seq in sorted(os.listdir(cls_dir)):
            seq_dir = os.path.join(cls_dir, seq)
            if os.path.isdir(seq_dir):
                frames = sorted([os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if f.lower().endswith(".jpg")])
                if frames:
                    clip_list.append((frames, class_map[cls]))
    return clip_list, classes

def load_clip_noaug(frame_paths, label):
    clip = []
    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_paths)-1)
        img_raw = tf.io.read_file(frame_paths[idx])
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        clip.append(img)
    clip = tf.stack(clip)
    label_onehot = tf.one_hot(label, depth=MAX_CLASSES)
    return clip, label_onehot

def tf_dataset(clip_list):
    ds = tf.data.Dataset.from_generator(
        lambda: ((np.array(frames, dtype=object), int(lbl)) for frames, lbl in clip_list),
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    ds = ds.map(lambda x,y: tf.py_function(load_clip_noaug, [x,y],[tf.float32,tf.float32]),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)
print(f"[INFO] Loaded model: {MODEL_PATH}")

# ---------------- LOAD TEST DATA ----------------
test_clips, class_names = get_test_clips()
print(f"[INFO] Test clips: {len(test_clips)}, Classes: {len(class_names)}")
test_ds = tf_dataset(test_clips)

# ---------------- EVALUATE ----------------
steps = math.ceil(len(test_clips)/BATCH_SIZE)
test_metrics = model.evaluate(test_ds, steps=steps, verbose=1)
print(f"[INFO] Test metrics: {test_metrics}")

# ---------------- SAVE JSON ----------------
test_json_path = os.path.join(os.path.dirname(MODEL_PATH), "test_metrics.json")
with open(test_json_path, "w") as f:
    json.dump({"loss": float(test_metrics[0]), "accuracy": float(test_metrics[1])}, f, indent=2)
print(f"[INFO] Test metrics saved: {test_json_path}")

# ---------------- PLOT ----------------
plt.figure(figsize=(6,4))
plt.bar(["Accuracy","Loss"], [float(test_metrics[1]), float(test_metrics[0])])
plt.title("Test Metrics")
plt.savefig(TEST_BAR_PLOT)
plt.show()
print(f"[INFO] Test bar plot saved: {TEST_BAR_PLOT}")
