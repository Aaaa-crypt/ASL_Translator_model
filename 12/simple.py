# ------------------------ training_model5_simplified.py ------------------------
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from PIL import Image
import matplotlib.pyplot as plt

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 6          # reduced from 12 to simplify sequences
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS = 10
MAX_CLASSES = 100
MODEL_PATH = "asl_seq_model_smart_v5.keras"
HISTORY_PATH = "history_seq_smart_v5.json"
TEST_BAR_PLOT = "test_metrics_bar_v5.png"

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
print("[INFO] Loading datasets...")
train_ds, class_names, train_steps = create_tf_dataset("train")
val_ds, _, val_steps = create_tf_dataset("val")
test_ds, _, test_steps = create_tf_dataset("test")
print(f"[INFO] Detected {len(class_names)} classes.")

# ---------------- MODEL ----------------
input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
inputs = layers.Input(shape=input_shape)

# Frozen MobileNetV2 backbone
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
backbone.trainable = False

x = layers.TimeDistributed(backbone)(inputs)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
x = layers.Bidirectional(
    layers.LSTM(64, return_sequences=False, 
                kernel_regularizer=tf.keras.regularizers.l2(1e-4))
)(x)
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
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

# ---------------- TRAIN ----------------
print("[INFO] Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks_list
)

# ---------------- SAVE HISTORY ----------------
with open(HISTORY_PATH, "w") as f:
    json.dump(history.history, f)
print(f"[INFO] Training history saved to {HISTORY_PATH}")

# ---------------- PLOT TRAINING GRAPHS ----------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_graphs_v5.png")
plt.show()
print("[INFO] Training graphs saved to training_graphs_v5.png")

# ---------------- EVALUATE ON TEST SET ----------------
print("[INFO] Evaluating on test set...")
test_metrics = model.evaluate(test_ds, steps=test_steps)
print(f"[INFO] Test Loss: {test_metrics[0]:.4f}, Test Accuracy: {test_metrics[1]:.4f}")

# ---------------- PLOT TEST METRICS BAR ----------------
plt.figure(figsize=(6,4))
plt.bar(["Accuracy","Loss"], [float(test_metrics[1]), float(test_metrics[0])], color=['green','red'])
plt.title("Test Metrics")
plt.ylabel("Value")
plt.ylim(0, 1)
plt.savefig(TEST_BAR_PLOT)
plt.show()
print(f"[INFO] Test metrics bar plot saved to {TEST_BAR_PLOT}")
print("✅ Done.")
