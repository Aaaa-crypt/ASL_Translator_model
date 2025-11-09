# ------------------------ train_from_sequences2.py ------------------------
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_partial"
FRAMES_PER_CLIP = 5        # can increase later
IMG_SIZE = (112, 112)
BATCH_SIZE = 8# ------------------------ train_from_sequences1.py (debug version) ------------------------
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import matplotlib.pyplot as plt

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_partial"
FRAMES_PER_CLIP = 5
IMG_SIZE = (112, 112)
BATCH_SIZE = 8
EPOCHS = 6                 # short test run
FRACTION = 0.10            # use 10% of data

MODEL_PATH = "asl_seq_model.keras"
HISTORY_PATH = "history_seq.json"

# ---------------- HELPERS ----------------
def load_sequences(split="train", class_names=None):
    split_dir = os.path.join(SEQUENCES_DIR, split)
    X, y = [], []

    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
    class_map = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        vids = sorted([
            d for d in os.listdir(cls_dir)
            if os.path.isdir(os.path.join(cls_dir, d))
        ])
        n_sample = max(1, int(len(vids) * FRACTION))
        vids = vids[:n_sample]

        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([
                f for f in os.listdir(full_vid_dir) if f.endswith(".jpg")
            ])[:FRAMES_PER_CLIP]
            if len(frames) < FRAMES_PER_CLIP:
                continue
            clip = []
            for f in frames:
                img = Image.open(os.path.join(full_vid_dir, f)).resize(IMG_SIZE)
                clip.append(np.array(img) / 255.0)
            X.append(clip)
            y.append(class_map[cls])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, class_names

# ---------------- LOAD DATA ----------------
print("[INFO] Loading training data...")
X_train, y_train, class_names = load_sequences("train")
print("[INFO] Loading validation data...")
X_val, y_val, _ = load_sequences("val", class_names=class_names)
print(f"[INFO] Detected {len(class_names)} classes.")
print(f"[INFO] Train shape: {X_train.shape}, Val shape: {X_val.shape}")

# ---------------- MODEL ----------------
input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
inputs = layers.Input(shape=input_shape, name="input_layer")

# Simplify CNN a bit for debugging
x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'))(inputs)
x = layers.TimeDistributed(layers.MaxPooling2D())(x)
x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'))(x)
x = layers.TimeDistributed(layers.MaxPooling2D())(x)
x = layers.TimeDistributed(layers.Flatten())(x)

x = layers.LSTM(64, return_sequences=False)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)

# ↓ Lower learning rate and simpler loss
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
]

# ---------------- TRAIN ----------------
print("[INFO] Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ---------------- SAVE HISTORY ----------------
with open(HISTORY_PATH, "w") as f:
    json.dump(history.history, f)
print(f"[INFO] Training history saved to {HISTORY_PATH}")

# ---------------- PLOT GRAPHS ----------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_seq.png")
plt.show()

print("[DONE] Model and training graphs saved.")

EPOCHS = 10
FRACTION = 0.10             # use 10% of data for quick test

MODEL_PATH = "asl_seq_model.keras"
HISTORY_PATH = "history_seq.json"

# ---------------- HELPERS ----------------
def load_sequences(split="train", class_names=None):
    """
    Load sequences as NumPy arrays and one-hot labels.
    Ensures all splits use the same class_names.
    """
    split_dir = os.path.join(SEQUENCES_DIR, split)
    X, y = [], []

    # consistent class order
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
    class_map = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        vids = sorted([
            d for d in os.listdir(cls_dir)
            if os.path.isdir(os.path.join(cls_dir, d))
        ])
        n_sample = max(1, int(len(vids) * FRACTION))
        vids = vids[:n_sample]

        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([
                f for f in os.listdir(full_vid_dir) if f.endswith(".jpg")
            ])[:FRAMES_PER_CLIP]
            if len(frames) < FRAMES_PER_CLIP:
                continue
            clip = []
            for f in frames:
                img = Image.open(os.path.join(full_vid_dir, f)).resize(IMG_SIZE)
                clip.append(np.array(img) / 255.0)
            X.append(clip)
            y.append(class_map[cls])

    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
    return X, y, class_names

# ---------------- LOAD DATA ----------------
print("[INFO] Loading training data...")
X_train, y_train, class_names = load_sequences("train")
print("[INFO] Loading validation data...")
X_val, y_val, _ = load_sequences("val", class_names=class_names)
print(f"[INFO] Detected {len(class_names)} classes.")

# ---------------- MODEL ----------------
input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
inputs = layers.Input(shape=input_shape, name="input_layer")

x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'))(inputs)
x = layers.TimeDistributed(layers.MaxPooling2D())(x)
x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'))(x)
x = layers.TimeDistributed(layers.MaxPooling2D())(x)
x = layers.TimeDistributed(layers.Flatten())(x)

x = layers.LSTM(128)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- CALLBACKS ----------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
]

# ---------------- TRAIN ----------------
print("[INFO] Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ---------------- SAVE HISTORY ----------------
with open(HISTORY_PATH, "w") as f:
    json.dump(history.history, f)
print(f"[INFO] Training history saved to {HISTORY_PATH}")

# ---------------- PLOT GRAPHS ----------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves_seq.png")
plt.show()

print("[DONE] Model and training graphs saved.")
