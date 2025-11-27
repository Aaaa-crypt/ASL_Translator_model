# ------------------------ train_from_sequences5.py ------------------------
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 6
IMG_SIZE = (96, 96)      # smaller = faster
BATCH_SIZE = 8
EPOCHS = 15
FRACTION = 0.5           # half dataset for testing
MAX_CLASSES = 200
PATIENCE = 5

MODEL_PATH = "asl_seq_model_smart_v3.keras"
HISTORY_PATH = "history_seq_smart_v3.json"

# ---------------- HELPERS ----------------
def load_sequence_paths(split="train", class_names=None, max_classes=None):
    """
    Returns a list of (clip_dir, class_index) tuples
    """
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        if max_classes:
            class_names = class_names[:max_classes]
    class_map = {c: i for i, c in enumerate(class_names)}

    paths = []
    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        n_sample = max(1, int(len(vids) * FRACTION))
        vids = vids[:n_sample]
        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            paths.append((full_vid_dir, class_map[cls]))
    return paths, class_names

def preprocess_clip(clip_dir):
    frames = sorted([f for f in os.listdir(clip_dir) if f.lower().endswith(".jpg")])[:FRAMES_PER_CLIP]
    if len(frames) < FRAMES_PER_CLIP:
        return None
    clip = []
    for f in frames:
        path = os.path.join(clip_dir, f)
        try:
            img = Image.open(path).convert("RGB").resize(IMG_SIZE)
            clip.append(np.array(img)/255.0)
        except:
            return None
    return np.stack(clip)

def tf_data_generator(paths, num_classes):
    def generator():
        for clip_dir, cls_idx in paths:
            clip = preprocess_clip(clip_dir)
            if clip is not None:
                yield clip, tf.keras.utils.to_categorical(cls_idx, num_classes)
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(FRAMES_PER_CLIP, *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    ).shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------- LOAD DATA ----------------
print("[INFO] Loading train paths...")
train_paths, class_names = load_sequence_paths("train", max_classes=MAX_CLASSES)
val_paths, _ = load_sequence_paths("val", class_names=class_names)
NUM_CLASSES = len(class_names)
print(f"[INFO] {len(train_paths)} train clips, {len(val_paths)} val clips, {NUM_CLASSES} classes.")

train_dataset = tf_data_generator(train_paths, NUM_CLASSES)
val_dataset = tf_data_generator(val_paths, NUM_CLASSES)

# ---------------- MODEL ----------------
inputs = layers.Input(shape=(FRAMES_PER_CLIP, *IMG_SIZE, 3))
cnn_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE,3))
cnn_base.trainable = False
x = layers.TimeDistributed(cnn_base)(inputs)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
x = layers.LSTM(128)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
]

# ---------------- TRAIN ----------------
print("[INFO] Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------- SAVE HISTORY ----------------
with open(HISTORY_PATH, "w") as f:
    json.dump(history.history, f)
print(f"[INFO] Training history saved to {HISTORY_PATH}")

# ---------------- PLOT GRAPHS ----------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_seq_smart_v3.png")
plt.show()
print("[DONE] Model and training graphs saved.")
