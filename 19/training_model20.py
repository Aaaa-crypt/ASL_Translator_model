# training_model20_fixed.py
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 40
MAX_CLASSES = 200
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\16"
FURTHER_INFO_DIR = os.path.join(OUTPUT_DIR, "further_info")
os.makedirs(FURTHER_INFO_DIR, exist_ok=True)

MODEL_BASE_PATH = os.path.join(OUTPUT_DIR, "asl_seq_model20")
HISTORY_BASE_PATH = os.path.join(OUTPUT_DIR, "history_seq20")
TEST_JSON_PATH = os.path.join(OUTPUT_DIR, "test20.json")
TEST_BAR_PLOT = os.path.join(OUTPUT_DIR, "test_bar_graph20.png")

# ---------------- HELPERS ----------------
def get_clip_paths_for_split(split="train"):
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
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
            frames = sorted([os.path.join(full_vid_dir, f) for f in os.listdir(full_vid_dir) if f.lower().endswith(".jpg")])
            if frames:
                clip_paths.append((frames, class_map[cls]))
    return clip_paths, class_names

train_clips, class_names_train = get_clip_paths_for_split("train")
val_clips, class_names_val = get_clip_paths_for_split("val")
test_clips, class_names_test = get_clip_paths_for_split("test")

if class_names_train[:len(class_names_val)] != class_names_val[:len(class_names_train)]:
    print("[WARN] Class name ordering or contents differ between splits.")

num_classes = len(class_names_train)
print(f"[INFO] Train clips: {len(train_clips)}, Val clips: {len(val_clips)}, Test clips: {len(test_clips)}")
print(f"[INFO] Using {num_classes} classes (MAX_CLASSES={MAX_CLASSES})")

# ---------------- DATA AUGMENTATION ----------------
def random_zoom_image(img, zoom_range=(0.96, 1.04)):
    """Pure TF zoom (no variables created inside tf.function)."""
    h, w = IMG_SIZE
    zoom_factor = tf.random.uniform([], zoom_range[0], zoom_range[1])
    new_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)
    img = tf.image.resize(img, (new_h, new_w))
    img = tf.image.resize_with_crop_or_pad(img, h, w)
    return img

def load_clip_py(frame_paths, label):
    frame_paths = [p.decode('utf-8') for p in frame_paths.numpy()]
    clip = []
    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_paths) - 1)
        img_raw = tf.io.read_file(frame_paths[idx])
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        # Mild augmentations
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.06)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = random_zoom_image(img)
        img = preprocess_input(img)
        clip.append(img)
    clip = tf.stack(clip)  # (T, H, W, C)
    label_onehot = tf.one_hot(label, depth=num_classes)
    return clip, label_onehot

def tf_load_clip(frame_paths, label):
    clip, lbl = tf.py_function(load_clip_py, [frame_paths, label], [tf.float32, tf.float32])
    clip.set_shape([FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3])
    lbl.set_shape([num_classes])
    return clip, lbl

def tf_load_clip_noaug(frame_paths, label):
    def _inner(fp, lbl):
        frame_paths_list = [p.decode("utf-8") for p in fp.numpy()]
        clip = []
        for i in range(FRAMES_PER_CLIP):
            idx = min(i, len(frame_paths_list) - 1)
            img_raw = tf.io.read_file(frame_paths_list[idx])
            img = tf.image.decode_jpeg(img_raw, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = preprocess_input(img)
            clip.append(img)
        clip = tf.stack(clip)
        lbl_oh = tf.one_hot(lbl, depth=num_classes)
        return clip, lbl_oh
    clip, lbl = tf.py_function(_inner, [frame_paths, label], [tf.float32, tf.float32])
    clip.set_shape([FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3])
    lbl.set_shape([num_classes])
    return clip, lbl

def create_tf_dataset_from_list(clip_list, repeat=False, augment=False):
    ds = tf.data.Dataset.from_generator(
        lambda: ((np.array(frames, dtype=object), int(lbl)) for frames, lbl in clip_list),
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    if augment:
        ds = ds.map(tf_load_clip, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=max(1024, len(clip_list)))
    else:
        ds = ds.map(tf_load_clip_noaug, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------- MODEL ----------------
def temporal_conv_block(x, filters=128, kernel_size=3):
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x

class TemporalAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units, activation='tanh')
        self.V = layers.Dense(1)

    def call(self, inputs):
        score = self.V(self.W(inputs))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

def build_model(num_classes, backbone_trainable=True, lr=1e-5):
    frames_input = layers.Input(shape=(FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3), name="frames")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False, weights='imagenet'
    )
    backbone.trainable = backbone_trainable
    x = layers.TimeDistributed(backbone)(frames_input)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = temporal_conv_block(x, filters=256, kernel_size=3)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = TemporalAttention(128)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(frames_input, outputs)
    model.compile(optimizer=optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------- DATASETS ----------------
train_ds = create_tf_dataset_from_list(train_clips, repeat=True, augment=True)
val_ds = create_tf_dataset_from_list(val_clips, repeat=False, augment=False)
test_ds = create_tf_dataset_from_list(test_clips, repeat=False, augment=False)

steps_per_epoch = math.ceil(len(train_clips)/BATCH_SIZE)
val_steps = math.ceil(len(val_clips)/BATCH_SIZE)

# ---------------- CLASS WEIGHTS ----------------
try:
    y_train = np.array([lbl for _, lbl in train_clips], dtype=int)
    class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_train)
    class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}
except Exception:
    class_weights = None
    print("[WARN] Could not compute class weights.")

# ---------------- CALLBACKS ----------------
def make_ckpt_path(phase="phase1"):
    return f"{MODEL_BASE_PATH}_{phase}.h5"

# ---------------- TRAINING ----------------
model = build_model(num_classes, backbone_trainable=False, lr=1e-4)
ckpt_phase1 = make_ckpt_path("phase1")
callbacks_phase1 = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint(ckpt_phase1, monitor='val_loss', save_best_only=True, save_weights_only=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

hist1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks_phase1,
    class_weight=class_weights
)

# Phase 2: Fine-tune backbone
for layer in model.layers:
    if isinstance(layer, layers.TimeDistributed):
        if hasattr(layer.layer, "trainable"):
            layer.layer.trainable = True

model.compile(optimizer=optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
ckpt_phase2 = make_ckpt_path("phase2")
callbacks_phase2 = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint(ckpt_phase2, monitor='val_loss', save_best_only=True, save_weights_only=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

hist2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks_phase2,
    class_weight=class_weights
)

# ---------------- EVALUATION ----------------
val_preds, val_labels = [], []
for x_batch, y_batch in val_ds:
    y_pred = model.predict(x_batch)
    val_preds.extend(np.argmax(y_pred, axis=1))
    val_labels.extend(np.argmax(y_batch.numpy(), axis=1))

metrics_info = {
    'precision': precision_score(val_labels, val_preds, average='weighted', zero_division=0),
    'recall': recall_score(val_labels, val_preds, average='weighted', zero_division=0),
    'f1_score': f1_score(val_labels, val_preds, average='weighted', zero_division=0)
}
with open(os.path.join(FURTHER_INFO_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics_info, f, indent=2)

# ---------------- TEST ----------------
best_ckpt = ckpt_phase2 if os.path.exists(ckpt_phase2) else ckpt_phase1
test_model = build_model(num_classes, backbone_trainable=True, lr=1e-5)
test_model.load_weights(best_ckpt)
test_metrics = test_model.evaluate(test_ds, steps=math.ceil(len(test_clips)/BATCH_SIZE), verbose=1)
with open(TEST_JSON_PATH, "w") as f:
    json.dump({"loss": float(test_metrics[0]), "accuracy": float(test_metrics[1])}, f, indent=2)

# ---------------- PLOTS ----------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(hist1.history.get('accuracy', []) + hist2.history.get('accuracy', []), label='Train')
plt.plot(hist1.history.get('val_accuracy', []) + hist2.history.get('val_accuracy', []), label='Val')
plt.title("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(hist1.history.get('loss', []) + hist2.history.get('loss', []), label='Train')
plt.plot(hist1.history.get('val_loss', []) + hist2.history.get('val_loss', []), label='Val')
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_graphs.png"))
plt.show()

plt.figure(figsize=(6,4))
plt.bar(["Accuracy","Loss"], [float(test_metrics[1]), float(test_metrics[0])])
plt.title("Test Metrics")
plt.savefig(TEST_BAR_PLOT)
plt.show()

print("✅ Done.")
