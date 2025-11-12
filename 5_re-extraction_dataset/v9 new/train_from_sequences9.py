# ------------------------ train_from_sequences9.py ------------------------
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
FRACTION = 0.5
MAX_CLASSES = 200

MODEL_PATH = "asl_seq_model_smart_v9.keras"
HISTORY_PATH = "history_seq_smart_v9.json"

# ---------------- HELPERS ----------------
def sequence_generator(split="train", class_names=None, max_classes=None):
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"❌ Split directory not found: {split_dir}")

    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        if max_classes:
            class_names = class_names[:max_classes]

    class_map = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        n_sample = max(1, int(len(vids) * FRACTION))
        vids = vids[:n_sample]

        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([f for f in os.listdir(full_vid_dir) if f.lower().endswith(".jpg")])

            clip = []
            for i in range(FRAMES_PER_CLIP):
                if i < len(frames):
                    path = os.path.join(full_vid_dir, frames[i])
                    try:
                        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
                        clip.append((np.array(img)/255.0).astype(np.float32))
                    except Exception as e:
                        print(f"[WARN] Could not load {path}: {e}")
                        clip.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32))
                else:
                    clip.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32))

            yield np.stack(clip), class_map[cls]

# ---------------- LOAD DATA USING TF.DATA ----------------
def load_tf_dataset(split="train", class_names=None, max_classes=None):
    dataset_list = list(sequence_generator(split, class_names, max_classes))
    if not dataset_list:
        raise ValueError(f"No data found for split {split}")

    if class_names is None:
        class_names = sorted([d for d in os.listdir(os.path.join(SEQUENCES_DIR, split)) 
                              if os.path.isdir(os.path.join(SEQUENCES_DIR, split, d))])
        if max_classes:
            class_names = class_names[:max_classes]

    num_classes = len(class_names)

    def gen():
        for x, y in dataset_list:
            yield x, tf.keras.utils.to_categorical(y, num_classes=num_classes)

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    )

    # Shuffle, batch, prefetch (no repeat)
    dataset = dataset.shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset, class_names

# ---------------- LOAD DATA ----------------
print("[INFO] Loading training data...")
train_ds, class_names = load_tf_dataset("train", max_classes=MAX_CLASSES)
print("[INFO] Loading validation data...")
val_ds, _ = load_tf_dataset("val", class_names=class_names)
print(f"[INFO] Detected {len(class_names)} classes (from training split).")

# ---------------- MODEL ----------------
input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
inputs = layers.Input(shape=input_shape)

backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
backbone.trainable = False

x = layers.TimeDistributed(backbone)(inputs)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
x = layers.LSTM(128)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(1e-4),
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
print("[INFO] Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
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
plt.savefig("training_curves_seq_smart_v9.png")
plt.show()

print("[DONE] Model and training graphs saved.")
