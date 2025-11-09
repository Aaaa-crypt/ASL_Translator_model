# ------------------------ training_model13_effnet_cv.py ------------------------
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 20
MAX_CLASSES = 100
N_FOLDS = 3

MODEL_BASE_PATH = "asl_seq_model_effnet_cv_fold"
HISTORY_BASE_PATH = "history_seq_effnet_cv_fold"

# ---------------- HELPERS ----------------
def get_clip_paths(max_classes=MAX_CLASSES):
    """Get all clips and labels from all class folders (no split)."""
    split_dir = os.path.join(SEQUENCES_DIR, "train")
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"❌ Train directory not found: {split_dir}")
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])[:max_classes]
    class_map = {c: i for i, c in enumerate(class_names)}
    clip_paths = []
    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([os.path.join(full_vid_dir, f) for f in os.listdir(full_vid_dir) if f.lower().endswith(".jpg")])
            if frames:
                clip_paths.append((frames, class_map[cls]))
    return clip_paths, class_names

def load_clip(frame_paths, label):
    clip = []
    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_paths) - 1)
        path = frame_paths[idx]
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        clip.append(img)
    clip = tf.stack(clip)
    label = tf.one_hot(label, depth=num_classes)
    return clip, label

def tf_load_clip(frame_paths, label):
    clip, lbl = tf.py_function(load_clip, [frame_paths, label], [tf.float32, tf.float32])
    clip.set_shape([FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3])
    lbl.set_shape([num_classes])
    return clip, lbl

def create_tf_dataset(clip_subset):
    dataset = tf.data.Dataset.from_generator(
        lambda: clip_subset,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    dataset = dataset.map(tf_load_clip, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(500).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    return dataset

# ---------------- MODEL CREATION ----------------
def build_model(num_classes):
    input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
    inputs = layers.Input(shape=input_shape)

    # ✅ Force RGB input for EfficientNetB0
    backbone = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )

    # Optional fine-tuning: freeze early layers
    for layer in backbone.layers[:150]:
        layer.trainable = False

    # TimeDistributed wrapper for video clips
    x = layers.TimeDistributed(backbone)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    # Sequence modeling via BiLSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------- CROSS-VALIDATION TRAINING ----------------
clip_paths, class_names = get_clip_paths()
num_classes = len(class_names)
print(f"[INFO] Loaded {len(clip_paths)} clips across {num_classes} classes.")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold = 1
cv_results = []

for train_index, val_index in kf.split(clip_paths):
    print(f"\n[INFO] Starting Fold {fold}/{N_FOLDS}...")
    train_subset = [clip_paths[i] for i in train_index]
    val_subset = [clip_paths[i] for i in val_index]

    train_ds = create_tf_dataset(train_subset)
    val_ds = create_tf_dataset(val_subset)
    steps_per_epoch = max(1, len(train_subset) // BATCH_SIZE)
    val_steps = max(1, len(val_subset) // BATCH_SIZE)

    model = build_model(num_classes)

    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ModelCheckpoint(f"{MODEL_BASE_PATH}{fold}.keras", monitor='val_loss', save_best_only=True),
        lr_scheduler
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks_list
    )

    # Save training history for this fold
    with open(f"{HISTORY_BASE_PATH}{fold}.json", "w") as f:
        json.dump(history.history, f)
    print(f"[INFO] History saved for fold {fold}.")

    # Record best val_accuracy
    best_val_acc = max(history.history['val_accuracy'])
    cv_results.append(best_val_acc)
    print(f"[RESULT] Fold {fold} best val_acc: {best_val_acc:.4f}")

    fold += 1

# ---------------- AGGREGATE RESULTS ----------------
mean_val_acc = np.mean(cv_results)
print(f"\n✅ Cross-validation complete. Mean val_acc across {N_FOLDS} folds: {mean_val_acc:.4f}")

with open("cv_summary_effnet.json", "w") as f:
    json.dump({"fold_val_accuracies": cv_results, "mean_val_accuracy": float(mean_val_acc)}, f)
print("[INFO] Cross-validation summary saved to cv_summary_effnet.json")
