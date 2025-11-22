# training_model21.py
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
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\20_simple_model\kaggle_dataset_mixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FURTHER_INFO_DIR = os.path.join(OUTPUT_DIR, "further_info")
os.makedirs(FURTHER_INFO_DIR, exist_ok=True)

MODEL_BASE_PATH = os.path.join(OUTPUT_DIR, "asl_seq_model_minimal")
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
steps = math.ceil(len(train_clips)/BATCH_SIZE)
val_steps = math.ceil(len(val_clips)/BATCH_SIZE)

# ---------------- MODEL ----------------
def build_minimal_model(num_classes, backbone_trainable=True, lr=1e-5):
    inp = layers.Input((FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3))
    base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base.trainable = backbone_trainable
    x = layers.TimeDistributed(base)(inp)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------- TRAIN PHASE 1 (frozen backbone) ----------------
model = build_minimal_model(num_classes, backbone_trainable=False, lr=1e-4)
ckpt1 = MODEL_BASE_PATH + "_phase1.weights.h5"
cb1 = [
    callbacks.ModelCheckpoint(ckpt1, save_weights_only=True, save_best_only=True),
    callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]
print("[INFO] Starting Phase 1 (frozen backbone)")
hist1 = model.fit(train_ds,
                  validation_data=val_ds,
                  steps_per_epoch=steps,
                  validation_steps=val_steps,
                  epochs=PHASE1_EPOCHS,
                  callbacks=cb1)

# Save Phase1 history
safe_hist1 = {k: [float(v) for v in vs] for k, vs in hist1.history.items()}
with open(HISTORY_PHASE1_PATH, "w") as f:
    json.dump(safe_hist1, f, indent=2)

# ---------------- TRAIN PHASE 2 (fine-tune backbone) ----------------
print("[INFO] Loading Phase1 weights and fine-tuning backbone")
model.load_weights(ckpt1)
model.trainable = True
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
ckpt2 = MODEL_BASE_PATH + "_phase2.weights.h5"
cb2 = [
    callbacks.ModelCheckpoint(ckpt2, save_weights_only=True, save_best_only=True),
    callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]
hist2 = model.fit(train_ds,
                  validation_data=val_ds,
                  steps_per_epoch=steps,
                  validation_steps=val_steps,
                  epochs=PHASE2_EPOCHS,
                  callbacks=cb2)

# ---------------- COMBINE & SAVE HISTORY ----------------
combined_history = {}
for k in set(hist1.history.keys()).union(hist2.history.keys()):
    vals = []
    vals.extend([float(v) for v in hist1.history.get(k, [])])
    vals.extend([float(v) for v in hist2.history.get(k, [])])
    combined_history[k] = vals
with open(HISTORY_PATH, "w") as f:
    json.dump(combined_history, f, indent=2)

# ---------------- VALIDATION METRICS ----------------
val_preds, val_true = [], []
for x, y in val_ds:
    p = model.predict(x)
    val_preds += list(np.argmax(p, axis=1))
    val_true += list(np.argmax(y.numpy(), axis=1))
metrics = {
    "precision": float(precision_score(val_true, val_preds, average='weighted', zero_division=0)),
    "recall": float(recall_score(val_true, val_preds, average='weighted', zero_division=0)),
    "f1": float(f1_score(val_true, val_preds, average='weighted', zero_division=0))
}
with open(METRICS_JSON_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------- TEST ----------------
best_weights_path = ckpt2 if os.path.exists(ckpt2) else ckpt1
test_model = build_minimal_model(num_classes, backbone_trainable=True, lr=1e-5)
if os.path.exists(best_weights_path):
    test_model.load_weights(best_weights_path)

test_metrics = test_model.evaluate(test_ds, steps=math.ceil(len(test_clips)/BATCH_SIZE), verbose=1)
with open(TEST_JSON_PATH, "w") as f:
    json.dump({"loss": float(test_metrics[0]), "accuracy": float(test_metrics[1])}, f, indent=2)

# ---------------- PLOTS ----------------
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

plt.figure(figsize=(4,4))
plt.bar(["Acc","Loss"], [float(test_metrics[1]), float(test_metrics[0])])
plt.title("Test Metrics")
plt.savefig(TEST_BAR_PLOT)
plt.show()

print("✅ Minimal training complete")
