# ------------------------ training_model12.py ------------------------
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from PIL import Image
import matplotlib.pyplot as plt

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 30
MAX_CLASSES = 100  # train only on first 100 for optimization
MODEL_PATH = "asl_seq_model_smart_v12.keras"
HISTORY_PATH = "history_seq_smart_v12.json"
TEST_JSON_PATH = "test_metrics_v12.json"
TEST_PLOT_PATH = "test_graphs_v12.png"

# ---------------- HELPERS ----------------
def get_clip_paths(split="train", max_classes=MAX_CLASSES):
    """Collect frame paths for each clip within each class folder."""
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
            frames = sorted([os.path.join(full_vid_dir, f) for f in os.listdir(full_vid_dir) if f.lower().endswith(".jpg")])
            if frames:
                clip_paths.append((frames, class_map[cls]))
    return clip_paths, class_names


def load_clip(frame_paths, label):
    """Load a fixed number of frames from a clip and apply augmentation."""
    clip = []
    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_paths) - 1)
        path = frame_paths[idx]
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0

        # ---- light data augmentation ----
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)

        clip.append(img)
    clip = tf.stack(clip)
    label = tf.one_hot(label, depth=num_classes)
    return clip, label


def tf_load_clip(frame_paths, label):
    """TF wrapper for clip loading."""
    clip, lbl = tf.py_function(load_clip, [frame_paths, label], [tf.float32, tf.float32])
    clip.set_shape([FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3])
    lbl.set_shape([num_classes])
    return clip, lbl


def create_tf_dataset(split="train"):
    """Build TensorFlow dataset with prefetch, shuffle, repeat."""
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
    dataset = dataset.shuffle(1000).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE).repeat()

    # show more unique samples per epoch
    steps = min(len(clip_paths) // BATCH_SIZE * 2, 300)
    return dataset, class_names, steps


# ---------------- LOAD DATA ----------------
print("[INFO] Loading training data...")
train_ds, class_names, train_steps = create_tf_dataset("train")
print("[INFO] Loading validation data...")
val_ds, _, val_steps = create_tf_dataset("val")
print("[INFO] Loading test data...")
test_ds, _, test_steps = create_tf_dataset("test")

print(f"[INFO] Detected {len(class_names)} classes (from training split).")
print(f"[INFO] Steps per epoch: {train_steps}, Validation steps: {val_steps}")

# ---------------- MODEL ----------------
input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
inputs = layers.Input(shape=input_shape)

# Pretrained backbone
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
# partially freeze
for layer in backbone.layers[:80]:
    layer.trainable = False

x = layers.TimeDistributed(backbone)(inputs)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(1e-4),  # balanced learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ---------------- CALLBACKS ----------------
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
    lr_scheduler
]

# ---------------- TRAIN ----------------
print("[INFO] Starting fine-tuning training...")
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

# ---------------- PLOT TRAIN/VAL GRAPHS ----------------
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
plt.savefig("training_graphs_v12.png")
plt.show()
print("[INFO] Training graphs saved to training_graphs_v12.png")

# ---------------- TEST EVALUATION ----------------
print("[INFO] Evaluating on test set...")
test_metrics = model.evaluate(test_ds, steps=test_steps, verbose=1)
test_results = {"loss": float(test_metrics[0]), "accuracy": float(test_metrics[1])}

# Save test metrics to JSON
with open(TEST_JSON_PATH, "w") as f:
    json.dump(test_results, f, indent=2)
print(f"[INFO] Test metrics saved to {TEST_JSON_PATH}")

# Plot test graph
plt.figure(figsize=(6,4))
plt.bar(["Test Accuracy", "Test Loss"], [test_results["accuracy"], test_results["loss"]], color=["orange", "blue"])
plt.title("Test Set Metrics")
plt.ylabel("Value")
plt.savefig(TEST_PLOT_PATH)
plt.show()
print(f"[INFO] Test graphs saved to {TEST_PLOT_PATH}")
