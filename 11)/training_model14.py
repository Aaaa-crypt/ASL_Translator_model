# training_model14.py
# Corrected cross-validation MobileNetV2 + BiLSTM training script
# - fixes dataset cache/repeat issues
# - two-phase training (freeze -> unfreeze)
# - saves per-fold histories, CV summary, test metrics + plots

import os
import json
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import KFold

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 25
PHASE1_EPOCHS = 5        # freeze backbone for these epochs
PHASE2_EPOCHS = EPOCHS - PHASE1_EPOCHS
MAX_CLASSES = 100
N_FOLDS = 3

MODEL_BASE_PATH = "asl_seq_model14"
HISTORY_BASE_PATH = "history_seq"
CV_SUMMARY_PATH = "cv_summary_mobilenet.json"
TEST_JSON_PATH = "test.json"
TEST_BAR_PLOT = "test_bar_graph.png"
TEST_LINE_PLOT = "test_line_graph.png"

OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- HELPERS ----------------
def get_clip_paths(split="train", max_classes=MAX_CLASSES):
    """Return list of (frames_list, label_index) for a split and the class list."""
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"❌ Split directory not found: {split_dir}")
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])[:max_classes]
    class_map = {c: i for i, c in enumerate(class_names)}
    clip_paths = []
    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([os.path.join(full_vid_dir, f)
                             for f in os.listdir(full_vid_dir)
                             if f.lower().endswith(".jpg")])
            if frames:
                clip_paths.append((frames, class_map[cls]))
    return clip_paths, class_names

def load_clip(frame_paths, label):
    """Loads FRAMES_PER_CLIP frames and returns (clip_tensor, one_hot_label)."""
    clip = []
    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_paths) - 1)
        path = frame_paths[idx]
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        clip.append(img)
    clip = tf.stack(clip)  # (T, H, W, C)
    label = tf.one_hot(label, depth=num_classes)
    return clip, label

def tf_load_clip(frame_paths, label):
    clip, lbl = tf.py_function(load_clip, [frame_paths, label], [tf.float32, tf.float32])
    clip.set_shape([FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3])
    lbl.set_shape([num_classes])
    return clip, lbl

def create_tf_dataset(clip_subset, repeat=False):
    """
    clip_subset: list of (frames_list, label)
    repeat: if True, dataset will be repeated (for training)
    Important: avoid cache().repeat() on infinite datasets; we limit dataset -> cache -> repeat.
    """
    ds = tf.data.Dataset.from_generator(
        lambda: clip_subset,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    ds = ds.map(tf_load_clip, num_parallel_calls=tf.data.AUTOTUNE)
    # shuffle with buffer = len(clip_subset) for good randomness (bounded)
    ds = ds.shuffle(buffer_size=max(512, len(clip_subset)))
    if repeat:
        # make finite before caching: take exact N samples, cache, then repeat
        ds = ds.take(len(clip_subset)).cache()
        ds = ds.repeat()
    # batch and prefetch
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------- MODEL ----------------
def build_model(num_classes, backbone_trainable=True, lr=1e-5):
    input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
    inputs = layers.Input(shape=input_shape)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    backbone.trainable = backbone_trainable

    x = layers.TimeDistributed(backbone)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------- CROSS-VALIDATION TRAINING ----------------
clip_paths_all, class_names = get_clip_paths("train")
num_classes = len(class_names)
print(f"[INFO] Loaded {len(clip_paths_all)} clips across {num_classes} classes.")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold = 1
cv_results = []
fold_histories = {}

for train_idx, val_idx in kf.split(clip_paths_all):
    print(f"\n[INFO] Starting Fold {fold}/{N_FOLDS}...")
    train_subset = [clip_paths_all[i] for i in train_idx]
    val_subset = [clip_paths_all[i] for i in val_idx]

    # datasets
    train_ds = create_tf_dataset(train_subset, repeat=True)
    val_ds = create_tf_dataset(val_subset, repeat=False)

    steps_per_epoch = math.ceil(len(train_subset) / BATCH_SIZE)
    val_steps = math.ceil(len(val_subset) / BATCH_SIZE)

    # Phase 1: frozen backbone
    model = build_model(num_classes, backbone_trainable=False, lr=1e-4)
    print("[INFO] Phase 1: backbone frozen")
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ckpt_path = f"{MODEL_BASE_PATH}_fold{fold}.keras"
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True),
        lr_scheduler
    ]
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    # Phase 2: unfreeze last layers and fine-tune
    print("[INFO] Phase 2: unfreezing backbone and fine-tuning")
    # load best weights from phase1 if saved
    if os.path.exists(ckpt_path):
        try:
            model = tf.keras.models.load_model(ckpt_path)
            print("[INFO] Loaded best model from Phase 1 checkpoint")
        except Exception:
            # if cannot load, continue with current model
            print("[WARN] Could not load ckpt; continuing with in-memory model")

    # Unfreeze backbone (optionally freeze early layers only)
    for layer in model.layers:
        # we want the backbone layers trainable -> easier to set entire backbone trainable
        # identify backbone by name 'mobilenetv2' in layer.name (TimeDistributed wrapper may prefix)
        pass
    # Simpler: rebuild model with trainable backbone but same weights where possible
    model = build_model(num_classes, backbone_trainable=True, lr=5e-6)
    # try to load weights from the saved checkpoint to preserve learned weights
    if os.path.exists(ckpt_path):
        try:
            model.load_weights(ckpt_path)
            print("[INFO] Loaded Phase 1 weights into fine-tune model")
        except Exception:
            print("[WARN] Could not load Phase1 weights into new model; continuing")

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    # Combine histories (concatenate lists)
    combined_history = {}
    for k in set(list(hist1.history.keys()) + list(hist2.history.keys())):
        combined_history[k] = hist1.history.get(k, []) + hist2.history.get(k, [])

    # Save per-fold history
    hist_path = f"{HISTORY_BASE_PATH}_fold{fold}.json"
    with open(hist_path, "w") as f:
        json.dump(combined_history, f)
    print(f"[INFO] History saved for fold {fold} -> {hist_path}")

    # record best val_acc
    best_val_acc = max(combined_history.get('val_accuracy', [0.0]))
    cv_results.append(float(best_val_acc))
    fold_histories[fold] = combined_history
    print(f"[RESULT] Fold {fold} best val_acc: {best_val_acc:.4f}")

    fold += 1

# ---------------- AGGREGATE RESULTS ----------------
mean_val_acc = float(np.mean(cv_results)) if cv_results else 0.0
cv_summary = {"fold_val_accuracies": cv_results, "mean_val_accuracy": mean_val_acc}
with open(CV_SUMMARY_PATH, "w") as f:
    json.dump(cv_summary, f, indent=2)
print(f"\n✅ Cross-validation complete. Mean val_acc: {mean_val_acc:.4f}")
print(f"[INFO] CV summary saved to {CV_SUMMARY_PATH}")

# Pick best fold model to evaluate on test set
if cv_results:
    best_fold = int(np.argmax(cv_results) + 1)  # folds are 1-indexed above
    best_model_path = f"{MODEL_BASE_PATH}_fold{best_fold}.keras"
    if os.path.exists(best_model_path):
        print(f"[INFO] Loading best fold model -> {best_model_path}")
        try:
            best_model = tf.keras.models.load_model(best_model_path)
        except Exception as e:
            print(f"[WARN] Could not load best model: {e}. Using final in-memory model.")
            best_model = model
    else:
        print("[WARN] Best model checkpoint missing; using final in-memory model.")
        best_model = model
else:
    best_model = model

# ---------------- TEST EVALUATION ----------------
print("[INFO] Loading full test dataset...")
test_clip_paths, _ = get_clip_paths("test")
test_ds = create_tf_dataset(test_clip_paths, repeat=False)
test_steps = math.ceil(len(test_clip_paths) / BATCH_SIZE)
print(f"[INFO] Test clips: {len(test_clip_paths)}, test_steps={test_steps}")

print("[INFO] Evaluating on test set...")
test_metrics = best_model.evaluate(test_ds, steps=test_steps, verbose=1)
test_results = {"loss": float(test_metrics[0]), "accuracy": float(test_metrics[1])}
with open(TEST_JSON_PATH, "w") as f:
    json.dump(test_results, f, indent=2)
print(f"[INFO] Test metrics saved to {TEST_JSON_PATH}")

# ---------------- PLOT TEST METRICS ----------------
# Bar plot
plt.figure(figsize=(6,4))
plt.bar(["Test Accuracy", "Test Loss"], [test_results["accuracy"], test_results["loss"]], color=["orange","blue"])
plt.title("Test Set Metrics")
plt.ylabel("Value")
plt.savefig(TEST_BAR_PLOT)
plt.close()
print(f"[INFO] Test bar plot saved to {TEST_BAR_PLOT}")

# Line plot: create a combined training plot using last fold history (or aggregate)
last_hist = fold_histories.get(best_fold, fold_histories.get(max(fold_histories.keys()), {}))
if last_hist:
    epochs_done = len(last_hist.get('accuracy', []))
    plt.figure(figsize=(9,4))
    plt.plot(range(1, epochs_done+1), last_hist.get('accuracy', []), label='Train Accuracy')
    plt.plot(range(1, epochs_done+1), last_hist.get('val_accuracy', []), label='Val Accuracy')
    plt.plot(range(1, epochs_done+1), last_hist.get('loss', []), label='Train Loss')
    plt.plot(range(1, epochs_done+1), last_hist.get('val_loss', []), label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics per Epoch (best fold)")
    plt.legend()
    plt.savefig(TEST_LINE_PLOT)
    plt.close()
    print(f"[INFO] Test line plot saved to {TEST_LINE_PLOT}")
else:
    print("[WARN] No per-epoch history available to plot line graph.")
