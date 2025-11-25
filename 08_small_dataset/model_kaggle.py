# model_kaggle.py
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\asl_sequences_mixed"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 8
PHASE1_EPOCHS = 12
PHASE2_EPOCHS = 12
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\08_small_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FURTHER_INFO_DIR = os.path.join(OUTPUT_DIR, "further_info")
os.makedirs(FURTHER_INFO_DIR, exist_ok=True)

MODEL_BASE_PATH = os.path.join(OUTPUT_DIR, "asl_seq_model")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history.json")
HISTORY_PHASE1_PATH = os.path.join(OUTPUT_DIR, "history_phase1.json")
TEST_JSON_PATH = os.path.join(OUTPUT_DIR, "test_metrics.json")
METRICS_JSON_PATH = os.path.join(FURTHER_INFO_DIR, "metrics.json")
TEST_BAR_PLOT = os.path.join(OUTPUT_DIR, "test_bar.png")
TRAINING_GRAPHS_PATH = os.path.join(OUTPUT_DIR, "training_graphs.png")

# ---------------- DATA LOADING ----------------
def get_clip_paths(split="train"):
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    class_map = {c: i for i, c in enumerate(class_names)}
    clip_paths = []
    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        for vid in vids:
            full_dir = os.path.join(cls_dir, vid)
            frames = sorted([os.path.join(full_dir, f) for f in os.listdir(full_dir) if f.lower().endswith(".jpg")])
            if frames:
                clip_paths.append((frames, class_map[cls]))
    return clip_paths, class_names

train_clips, class_names = get_clip_paths("train")
val_clips, _ = get_clip_paths("val")
test_clips, _ = get_clip_paths("test")
num_classes = len(class_names)

print(f"[INFO] Train: {len(train_clips)}, Val: {len(val_clips)}, Test: {len(test_clips)}, Classes: {num_classes}")

# ---------------- AUGMENTATION ----------------
def random_zoom_image(img, zoom_range=(0.95, 1.05)):
    h, w = IMG_SIZE
    z = np.random.uniform(*zoom_range)
    new_h = tf.cast(tf.round(h * z), tf.int32)
    new_w = tf.cast(tf.round(w * z), tf.int32)
    img = tf.image.resize(img, (h, w))
    scaled = tf.image.resize(img, (new_h, new_w))
    scaled = tf.image.resize_with_crop_or_pad(scaled, h, w)
    return tf.image.resize(scaled, IMG_SIZE)

# ---------------- DATASET ----------------
def make_dataset(clips, aug=False, repeat=False):
    def gen():
        for frames, lbl in clips:
            imgs = []
            for i in range(FRAMES_PER_CLIP):
                idx = min(i, len(frames) - 1)
                img = tf.io.read_file(frames[idx])
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, IMG_SIZE)
                if aug:
                    img = tf.image.random_flip_left_right(img)
                    img = tf.image.random_brightness(img, 0.06)
                    img = tf.image.random_contrast(img, 0.9, 1.1)
                    img = random_zoom_image(img, (0.96, 1.04))
                img = preprocess_input(tf.cast(img, tf.float32))
                imgs.append(img)
            imgs_tensor = tf.stack(imgs)
            lbl_tensor = tf.one_hot(lbl, depth=num_classes)
            yield imgs_tensor, lbl_tensor

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3), tf.float32),
            tf.TensorSpec((num_classes,), tf.float32)
        )
    )

    if repeat:
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_clips, aug=True, repeat=True)
val_ds = make_dataset(val_clips, aug=False)
test_ds = make_dataset(test_clips, aug=False)

steps = math.ceil(len(train_clips) / BATCH_SIZE) if len(train_clips) > 0 else 1
val_steps = math.ceil(len(val_clips) / BATCH_SIZE) if len(val_clips) > 0 else 1

# ---------------- MODEL ----------------
def temporal_conv_block(x, filters=128):
    x = layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
    return layers.BatchNormalization()(x)

class TemporalAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units, activation="tanh")
        self.V = layers.Dense(1)
    def call(self, x):
        w = tf.nn.softmax(self.V(self.W(x)), axis=1)
        return tf.reduce_sum(w * x, axis=1)

def build_model(num_classes, backbone_trainable=True, lr=1e-5):
    inp = layers.Input((FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3))
    base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base.trainable = backbone_trainable
    x = layers.TimeDistributed(base)(inp)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = temporal_conv_block(x, 256)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = TemporalAttention(128)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ---------------- TRAIN PHASE 1 ----------------
model = build_model(num_classes, backbone_trainable=False, lr=1e-4)
ckpt1 = MODEL_BASE_PATH + "_phase1.weights.h5"
if os.path.exists(ckpt1):
    try: os.remove(ckpt1)
    except Exception: pass

cb1 = [
    callbacks.ModelCheckpoint(ckpt1, save_weights_only=True, save_best_only=True),
    callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

print("[INFO] Starting Phase 1 (frozen backbone)")
hist1 = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps,
    validation_steps=val_steps,
    epochs=PHASE1_EPOCHS,
    callbacks=cb1
)

safe_hist1 = {k: [float(v) for v in vs] for k, vs in hist1.history.items()}
try:
    with open(HISTORY_PHASE1_PATH, "w") as f:
        json.dump(safe_hist1, f, indent=2)
    print(f"[INFO] Phase 1 history saved to {HISTORY_PHASE1_PATH}")
except Exception as e:
    print(f"[WARN] Could not save Phase1 history: {e}")

# ---------------- LOAD BEST PHASE 1 WEIGHTS BEFORE PHASE 2 ----------------
print("[INFO] Loading best Phase 1 weights for fine-tuning...")
try:
    model.load_weights(ckpt1)
    print("[INFO] Successfully loaded Phase 1 best weights.")
except Exception as e:
    print(f"[WARN] Could not load Phase 1 best weights: {e}")

# ---------------- TRAIN PHASE 2 ----------------
print("[INFO] Phase 2: fine-tuning (unfreeze backbone)")
model.trainable = True
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

ckpt2 = MODEL_BASE_PATH + "_phase2.weights.h5"
if os.path.exists(ckpt2):
    try: os.remove(ckpt2)
    except Exception: pass

cb2 = [
    callbacks.ModelCheckpoint(ckpt2, save_weights_only=True, save_best_only=True),
    callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

hist2 = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps,
    validation_steps=val_steps,
    epochs=PHASE2_EPOCHS,
    callbacks=cb2
)

# ---------------- COMBINE & SAVE HISTORIES ----------------
combined_history = {}
for k in set(hist1.history.keys()).union(hist2.history.keys()):
    vals = []
    vals.extend([float(v) for v in hist1.history.get(k, [])])
    vals.extend([float(v) for v in hist2.history.get(k, [])])
    combined_history[k] = vals

try:
    with open(HISTORY_PATH, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"[INFO] Combined history saved to {HISTORY_PATH}")
except Exception as e:
    print(f"[WARN] Could not save combined history: {e}")

# ---------------- VALIDATION METRICS ----------------
val_preds, val_true = [], []
for x, y in val_ds:
    p = model.predict(x)
    val_preds += list(np.argmax(p, axis=1))
    val_true  += list(np.argmax(y.numpy(), axis=1))

metrics = {
    "precision": float(precision_score(val_true, val_preds, average="weighted", zero_division=0)),
    "recall": float(recall_score(val_true, val_preds, average="weighted", zero_division=0)),
    "f1": float(f1_score(val_true, val_preds, average="weighted", zero_division=0))
}
try:
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Validation metrics saved to {METRICS_JSON_PATH}")
except Exception as e:
    print(f"[WARN] Could not save validation metrics: {e}")

# ---------------- TEST ----------------
best_weights_path = ckpt2 if os.path.exists(ckpt2) else ckpt1

test_model = build_model(num_classes, backbone_trainable=True, lr=1e-5)
if os.path.exists(best_weights_path):
    try:
        test_model.load_weights(best_weights_path)
        print(f"[INFO] Loaded best weights from {best_weights_path} for testing.")
    except Exception as e:
        print(f"[WARN] Could not load best weights ({best_weights_path}): {e}")
else:
    print("[WARN] No saved weights found; running test with current model instance")

test_metrics = test_model.evaluate(
    test_ds,
    steps=math.ceil(len(test_clips) / BATCH_SIZE) if len(test_clips) > 0 else 1,
    verbose=1
)

try:
    with open(TEST_JSON_PATH, "w") as f:
        json.dump({"loss": float(test_metrics[0]), "accuracy": float(test_metrics[1])}, f, indent=2)
    print(f"[INFO] Test metrics saved to {TEST_JSON_PATH}")
except Exception as e:
    print(f"[WARN] Could not save test metrics: {e}")

# ---------------- PLOTS ----------------
try:
    if combined_history:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(combined_history.get('accuracy', []), label='Train')
        plt.plot(combined_history.get('val_accuracy', []), label='Val')
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(combined_history.get('loss', []), label='Train')
        plt.plot(combined_history.get('val_loss', []), label='Val')
        plt.title("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(TRAINING_GRAPHS_PATH)
        print(f"[INFO] Training graphs saved to {TRAINING_GRAPHS_PATH}")

    plt.figure(figsize=(4,4))
    test_acc = float(test_metrics[1]) if isinstance(test_metrics, (list, tuple)) and len(test_metrics) > 1 else float(test_metrics)
    test_loss = float(test_metrics[0]) if isinstance(test_metrics, (list, tuple)) else 0.0
    plt.bar(["Acc","Loss"], [test_acc, test_loss])
    plt.title("Test Metrics")
    plt.savefig(TEST_BAR_PLOT)
    plt.show()

    print(f"[INFO] Test bar plot saved to {TEST_BAR_PLOT}")
except Exception as e:
    print(f"[WARN] Could not create/save plots: {e}")

print("✅ Done")
