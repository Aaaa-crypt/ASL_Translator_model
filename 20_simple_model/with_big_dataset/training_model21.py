# training_model21.py
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
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\20_simple_model"
FURTHER_INFO_DIR = os.path.join(OUTPUT_DIR, "further_info")
os.makedirs(FURTHER_INFO_DIR, exist_ok=True)

MODEL_BASE_PATH = os.path.join(OUTPUT_DIR, "model22")
HISTORY_BASE_PATH = os.path.join(OUTPUT_DIR, "history")
HISTORY_PHASE1_PATH = HISTORY_BASE_PATH + "phase1.json"
HISTORY_PHASE2_PATH = HISTORY_BASE_PATH + "phase2.json"
HISTORY_FULL_PATH = HISTORY_BASE_PATH + "history_full.json"
TEST_JSON_PATH = os.path.join(OUTPUT_DIR, "test.json")
TEST_BAR_PLOT = os.path.join(OUTPUT_DIR, "test_bar.png")

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
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.06)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = random_zoom_image(img)
        img = preprocess_input(img)
        clip.append(img)
    clip = tf.stack(clip)
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
def build_model(num_classes, backbone_trainable=True, lr=1e-5):
    frames_input = layers.Input(shape=(FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3), name="frames")
    
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False, weights='imagenet'
    )
    backbone.trainable = backbone_trainable
    x = layers.TimeDistributed(backbone)(frames_input)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(frames_input, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------- DATASETS ----------------
train_ds = create_tf_dataset_from_list(train_clips, repeat=True, augment=True)
val_ds = create_tf_dataset_from_list(val_clips, repeat=False, augment=False)
test_ds = create_tf_dataset_from_list(test_clips, repeat=False, augment=False)

steps_per_epoch = math.ceil(len(train_clips)/BATCH_SIZE) if len(train_clips) > 0 else 1
val_steps = math.ceil(len(val_clips)/BATCH_SIZE) if len(val_clips) > 0 else 1

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
    return f"{MODEL_BASE_PATH}_{phase}.weights.h5"

# ---------------- TRAINING ----------------
print("[INFO] Building model (phase1, frozen backbone)...")
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
    class_weight=class_weights,
    verbose=1
)

# Save phase1 history
try:
    safe_h1 = {k: [float(v) for v in vs] for k, vs in hist1.history.items()}
    with open(HISTORY_PHASE1_PATH, "w") as f:
        json.dump(safe_h1, f, indent=2)
    print(f"[INFO] Phase1 history saved to {HISTORY_PHASE1_PATH}")
except Exception as e:
    print(f"[WARN] Could not save Phase1 history: {e}")

# Phase 2: Fine-tune backbone
print("[INFO] Unfreezing backbone and recompiling for fine-tuning...")
for layer in model.layers:
    if isinstance(layer, layers.TimeDistributed):
        if hasattr(layer.layer, "trainable"):
            layer.layer.trainable = True

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
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
    class_weight=class_weights,
    verbose=1
)

# Save phase2 history
try:
    safe_h2 = {k: [float(v) for v in vs] for k, vs in hist2.history.items()}
    with open(HISTORY_PHASE2_PATH, "w") as f:
        json.dump(safe_h2, f, indent=2)
    print(f"[INFO] Phase2 history saved to {HISTORY_PHASE2_PATH}")
except Exception as e:
    print(f"[WARN] Could not save Phase2 history: {e}")

# ---------------- COMBINE & SAVE FULL HISTORY ----------------
combined_history = {}
for k in set(hist1.history.keys()).union(hist2.history.keys()):
    vals = []
    vals.extend([float(v) for v in hist1.history.get(k, [])])
    vals.extend([float(v) for v in hist2.history.get(k, [])])
    combined_history[k] = vals

try:
    with open(HISTORY_FULL_PATH, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"[INFO] Full combined history saved to {HISTORY_FULL_PATH}")
except Exception as e:
    print(f"[WARN] Could not save full history: {e}")

# ---------------- EVALUATION ----------------
print("[INFO] Running validation predictions for metrics...")
val_preds, val_labels = [], []
for x_batch, y_batch in val_ds:
    y_pred = model.predict(x_batch, verbose=0)
    val_preds.extend(np.argmax(y_pred, axis=1))
    val_labels.extend(np.argmax(y_batch.numpy(), axis=1))

metrics_info = {
    'precision': float(precision_score(val_labels, val_preds, average='weighted', zero_division=0)),
    'recall': float(recall_score(val_labels, val_preds, average='weighted', zero_division=0)),
    'f1_score': float(f1_score(val_labels, val_preds, average='weighted', zero_division=0))
}
try:
    with open(os.path.join(FURTHER_INFO_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_info, f, indent=2)
    print(f"[INFO] Validation metrics saved to {os.path.join(FURTHER_INFO_DIR, 'metrics.json')}")
except Exception as e:
    print(f"[WARN] Could not save validation metrics: {e}")

# ---------------- TEST ----------------
best_ckpt = ckpt_phase2 if os.path.exists(ckpt_phase2) else ckpt_phase1
test_model = build_model(num_classes, backbone_trainable=True, lr=1e-5)
if os.path.exists(best_ckpt):
    try:
        test_model.load_weights(best_ckpt)
        print(f"[INFO] Loaded best weights from {best_ckpt} for testing.")
    except Exception as e:
        print(f"[WARN] Could not load best weights ({best_ckpt}): {e}")
else:
    print("[WARN] No checkpoint found, test will use current model weights (may be untrained).")

test_metrics = test_model.evaluate(test_ds, steps=math.ceil(len(test_clips)/BATCH_SIZE) if len(test_clips)>0 else 1, verbose=1)
try:
    with open(TEST_JSON_PATH, "w") as f:
        json.dump({"loss": float(test_metrics[0]), "accuracy": float(test_metrics[1])}, f, indent=2)
    print(f"[INFO] Test metrics saved to {TEST_JSON_PATH}")
except Exception as e:
    print(f"[WARN] Could not save test metrics: {e}")

# ---------------- PLOTS ----------------
try:
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(combined_history.get('accuracy', []), label='Train')
    plt.plot(combined_history.get('val_accuracy', []), label='Val')
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(combined_history.get('loss', []), label='Train')
    plt.plot(combined_history.get('val_loss', []), label='Val')
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    training_graph_path = os.path.join(OUTPUT_DIR, "training_graphs.png")
    plt.savefig(training_graph_path)
    print(f"[INFO] Training graphs saved to {training_graph_path}")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(["Accuracy","Loss"], [float(test_metrics[1]), float(test_metrics[0])])
    plt.title("Test Metrics")
    plt.savefig(TEST_BAR_PLOT)
    print(f"[INFO] Test bar plot saved to {TEST_BAR_PLOT}")
    plt.close()
except Exception as e:
    print(f"[WARN] Could not create or save plots: {e}")

print("✅ Done.")
