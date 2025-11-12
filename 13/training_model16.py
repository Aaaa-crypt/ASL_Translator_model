# training_model16.py
# MobileNetV2 + BiLSTM training script
# Uses up to 200 classes from the full dataset, with proper preprocessing and augmentation.

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import KFold
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20
N_FOLDS = 3
MAX_CLASSES = 200

MODEL_BASE_PATH = "asl_seq_model16"
HISTORY_BASE_PATH = "history_seq16"
CV_SUMMARY_PATH = "cv_summary_mobilenet16.json"
TEST_JSON_PATH = "test16.json"
TEST_BAR_PLOT = "test_bar_graph16.png"

# ---------------- HELPERS ----------------
def get_clip_paths(split="train"):
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"❌ Split directory not found: {split_dir}")
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    if MAX_CLASSES:
        class_names = class_names[:MAX_CLASSES]
    class_map = {c: i for i, c in enumerate(class_names)}
    clip_paths = []
    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([
                os.path.join(full_vid_dir, f)
                for f in os.listdir(full_vid_dir)
                if f.lower().endswith(".jpg")
            ])
            if frames:
                clip_paths.append((frames, class_map[cls]))
    return clip_paths, class_names

clip_paths_all, class_names = get_clip_paths("train")
num_classes = len(class_names)

# ---------------- DATA LOADING ----------------
def load_clip(frame_paths, label):
    clip = []
    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_paths) - 1)
        img = tf.io.read_file(frame_paths[idx])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = preprocess_input(img)
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
    ds = tf.data.Dataset.from_generator(
        lambda: clip_subset,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    ds = ds.map(tf_load_clip, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=max(512, len(clip_subset)))
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------- MODEL ----------------
def build_model(num_classes, backbone_trainable=True, lr=1e-5):
    inputs = layers.Input(shape=(FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3))
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
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------- CROSS-VALIDATION TRAINING ----------------
print(f"[INFO] Loaded {len(clip_paths_all)} clips across {num_classes} classes")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
cv_results = []
fold_histories = {}

for fold, (train_idx, val_idx) in enumerate(kf.split(clip_paths_all), start=1):
    print(f"\n[INFO] Starting Fold {fold}/{N_FOLDS}...")
    train_subset = [clip_paths_all[i] for i in train_idx]
    val_subset = [clip_paths_all[i] for i in val_idx]

    train_ds = create_tf_dataset(train_subset, repeat=True)
    val_ds = create_tf_dataset(val_subset, repeat=False)

    steps_per_epoch = math.ceil(len(train_subset) / BATCH_SIZE)
    val_steps = math.ceil(len(val_subset) / BATCH_SIZE)

    # Phase 1: frozen backbone
    model = build_model(num_classes, backbone_trainable=False, lr=1e-4)
    print("[INFO] Phase 1: backbone frozen")
    ckpt_path = f"{MODEL_BASE_PATH}_fold{fold}.keras"
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
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

    # Phase 2: fine-tune
    print("[INFO] Phase 2: unfreezing backbone and fine-tuning")
    model = build_model(num_classes, backbone_trainable=True, lr=1e-5)
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    combined_history = {k: hist1.history.get(k, []) + hist2.history.get(k, []) 
                        for k in set(hist1.history.keys()).union(hist2.history.keys())}

    with open(f"{HISTORY_BASE_PATH}_fold{fold}.json", "w") as f:
        json.dump(combined_history, f)

    best_val_acc = max(combined_history.get('val_accuracy', [0.0]))
    cv_results.append(best_val_acc)
    fold_histories[fold] = combined_history
    print(f"[RESULT] Fold {fold} best val_acc: {best_val_acc:.4f}")

# ---------------- AGGREGATE RESULTS ----------------
mean_val_acc = float(np.mean(cv_results)) if cv_results else 0.0
with open(CV_SUMMARY_PATH, "w") as f:
    json.dump({"fold_val_accuracies": cv_results, "mean_val_accuracy": mean_val_acc}, f, indent=2)
print(f"\n✅ Cross-validation complete. Mean val_acc: {mean_val_acc:.4f}")

# ---------------- TEST ----------------
print("[INFO] Loading full test dataset...")
test_clip_paths, _ = get_clip_paths("test")
test_ds = create_tf_dataset(test_clip_paths, repeat=False)
test_steps = math.ceil(len(test_clip_paths) / BATCH_SIZE)

best_fold = int(np.argmax(cv_results) + 1)
best_model_path = f"{MODEL_BASE_PATH}_fold{best_fold}.keras"
model = tf.keras.models.load_model(best_model_path)
test_metrics = model.evaluate(test_ds, steps=test_steps, verbose=1)

with open(TEST_JSON_PATH, "w") as f:
    json.dump({"loss": float(test_metrics[0]), "accuracy": float(test_metrics[1])}, f, indent=2)
print(f"[INFO] Test metrics saved to {TEST_JSON_PATH}")

# ---------------- TRAINING GRAPHS ----------------
hist = fold_histories.get(best_fold, {})
if hist:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.get('accuracy', []), label='Train Accuracy', color='orange')
    plt.plot(hist.get('val_accuracy', []), label='Validation Accuracy', color='blue')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.get('loss', []), label='Train Loss', color='orange')
    plt.plot(hist.get('val_loss', []), label='Validation Loss', color='blue')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_graphs16.png")
    plt.show()
    print("[INFO] Training graphs saved to training_graphs16.png")
else:
    print("[WARN] No history found for best fold — skipping training plots.")

# ---------------- TEST BAR GRAPH ----------------
plt.figure(figsize=(6, 4))
plt.bar(["Test Accuracy", "Test Loss"],
        [test_metrics[1], test_metrics[0]],
        color=["orange", "blue"])
plt.title("Test Set Metrics")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(TEST_BAR_PLOT)
plt.show()
print(f"[INFO] Test bar graph saved to {TEST_BAR_PLOT}")
