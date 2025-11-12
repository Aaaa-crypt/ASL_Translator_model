# ------------------------ training_model2.py ------------------------
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 5
IMG_SIZE = (96, 96)
BATCH_SIZE = 8
EPOCHS = 15
FRACTION = 1.0  # use full dataset for better accuracy
MODEL_PATH = "asl_seq_model_smart_v11_finetuned.keras"
HISTORY_PATH = "history_seq_smart_v11_finetuned.json"
GRAPH_PATH = "training_graphs_v11_finetuned.png"

# ---------------- HELPERS ----------------
def get_clip_paths(split="train", class_names=None):
    """Return list of tuples: (list of frame paths, class_index)"""
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"❌ Split directory not found: {split_dir}")

    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
    class_map = {c: i for i, c in enumerate(class_names)}
    clip_paths = []

    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([
            d for d in os.listdir(cls_dir)
            if os.path.isdir(os.path.join(cls_dir, d))
        ])
        n_sample = max(1, int(len(vids) * FRACTION))
        vids = vids[:n_sample]

        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([
                os.path.join(full_vid_dir, f)
                for f in os.listdir(full_vid_dir)
                if f.lower().endswith(".jpg")
            ])
            if not frames:
                continue
            clip_paths.append((frames, class_map[cls]))
    return clip_paths, class_names

def load_clip(frame_paths, label):
    """Load a clip (list of frame paths) and convert to tensor."""
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
    clip.set_shape([FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3])  # fix shape
    lbl.set_shape([num_classes])
    return clip, lbl

# ---------------- LOAD DATA ----------------
def create_tf_dataset(split="train", class_names=None):
    clip_paths, class_names = get_clip_paths(split, class_names)
    global num_classes
    num_classes = len(class_names)

    dataset = tf.data.Dataset.from_generator(
        lambda: clip_paths,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    dataset = dataset.map(tf_load_clip, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).repeat()
    steps = max(1, len(clip_paths) // BATCH_SIZE)
    return dataset, class_names, steps

# ---------------- LOAD DATA ----------------
print("[INFO] Loading training data...")
train_ds, class_names, train_steps = create_tf_dataset("train")
print("[INFO] Loading validation data...")
val_ds, _, val_steps = create_tf_dataset("val", class_names=class_names)
print(f"[INFO] Detected {len(class_names)} classes (from training split).")

# ---------------- MODEL ----------------
input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
inputs = layers.Input(shape=input_shape)

# Base MobileNetV2 backbone
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Unfreeze last 20% of layers for fine-tuning
fine_tune_at = int(len(backbone.layers) * 0.8)
for layer in backbone.layers[:fine_tune_at]:
    layer.trainable = False
for layer in backbone.layers[fine_tune_at:]:
    layer.trainable = True

x = layers.TimeDistributed(backbone)(inputs)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
x = layers.LSTM(128, return_sequences=False)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
]

# ---------------- TRAIN ----------------
print("[INFO] Starting fine-tuning training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks
)

# ---------------- SAVE HISTORY ----------------
with open(HISTORY_PATH, "w") as f:
    json.dump(history.history, f)
print(f"[INFO] Training history saved to {HISTORY_PATH}")

# ---------------- PLOT GRAPHS ----------------
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(GRAPH_PATH)
plt.close()

print(f"[INFO] Training graphs saved to {GRAPH_PATH}")
