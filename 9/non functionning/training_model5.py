# ------------------------ training_model5.py ------------------------
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
MAX_CLASSES = 100
MODEL_BASE_PATH = "asl_seq_model_smart_v5"
HISTORY_PATH = f"history_seq_smart_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# ---------------- HELPERS ----------------
def get_clip_paths(split="train", max_classes=MAX_CLASSES):
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"❌ Split directory not found: {split_dir}")

    class_names = sorted([d for d in os.listdir(split_dir) 
                          if os.path.isdir(os.path.join(split_dir, d))])[:max_classes]
    class_map = {c: i for i, c in enumerate(class_names)}
    clip_paths = []

    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        for vid_dir in vids:
            full_vid_dir = os.path.join(cls_dir, vid_dir)
            frames = sorted([os.path.join(full_vid_dir, f) 
                             for f in os.listdir(full_vid_dir) if f.lower().endswith(".jpg")])
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

# ---------------- CREATE DATASETS ----------------
def create_tf_dataset(split="train"):
    clip_paths, class_names = get_clip_paths(split)
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
    dataset = dataset.shuffle(500).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE).repeat()
    steps = max(1, len(clip_paths) // BATCH_SIZE)
    return dataset, class_names, steps

# ---------------- LOAD DATA ----------------
print("[INFO] Loading training data...")
train_ds, class_names_train, train_steps = create_tf_dataset("train")
print("[INFO] Loading validation data...")
val_ds, class_names_val, val_steps = create_tf_dataset("val")

# ---------------- DIAGNOSTIC PRINT ----------------
print("✅ Running diagnostic class check...")
print(f"[DEBUG] First 10 TRAIN classes:\n{class_names_train[:10]}")
print(f"[DEBUG] First 10 VAL classes:\n{class_names_val[:10]}")
print(f"[DEBUG] TRAIN class count: {len(class_names_train)}")
print(f"[DEBUG] VAL class count: {len(class_names_val)}")
if class_names_train != class_names_val:
    print("❌ WARNING: TRAIN and VAL class lists do not match exactly!")
else:
    print("✅ Diagnostic complete: TRAIN and VAL class lists match.")

# ---------------- MODEL ----------------
input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
inputs = layers.Input(shape=input_shape)

# MobileNetV2 backbone
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
backbone.trainable = False  # Phase 1 frozen

x = layers.TimeDistributed(backbone)(inputs)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names_train), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ---------------- CALLBACKS ----------------
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    callbacks.ModelCheckpoint(f"{MODEL_BASE_PATH}.keras", monitor='val_loss', save_best_only=True),
    lr_scheduler
]

# ---------------- TRAIN PHASE 1 ----------------
print("[INFO] Starting Phase 1 training (backbone frozen)...")
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks_list
)

# ---------------- PHASE 2: UNFREEZE LAST 30 BACKBONE LAYERS ----------------
print("[INFO] Phase 1 done. Unfreezing last 30 layers of backbone and fine-tuning...")
for layer in backbone.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(1e-5),  # lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks_list
)

# ---------------- SAVE HISTORY ----------------
full_history = {
    "phase1": history_phase1.history,
    "phase2": history_phase2.history
}

with open(HISTORY_PATH, "w") as f:
    json.dump(full_history, f)
print(f"[INFO] Training history saved to {HISTORY_PATH}")

# ---------------- PLOT GRAPHS ----------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_phase1.history['accuracy'] + history_phase2.history['accuracy'], label='train acc')
plt.plot(history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'], label='val acc')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_phase1.history['loss'] + history_phase2.history['loss'], label='train loss')
plt.plot(history_phase1.history['val_loss'] + history_phase2.history['val_loss'], label='val loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(f"{MODEL_BASE_PATH}_training_graphs.png")
plt.show()
print(f"[INFO] Training graphs saved to {MODEL_BASE_PATH}_training_graphs.png")
