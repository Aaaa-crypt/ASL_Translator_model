# training_model4.py
import os
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt

# -------- CONFIG --------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
EPOCHS = 10
MAX_CLASSES = 100        # use first 100 classes for quicker iteration
MODEL_PATH = "asl_seq_model_smart_v4.keras"
CLASS_NAMES_PATH = "class_names_v4.json"
HISTORY_PATH = "history_seq_smart_v4.json"
PATIENCE_ES = 8          # EarlyStopping patience (you can increase)
PATIENCE_RLR = 3         # ReduceLROnPlateau patience

# Finetune settings: freeze backbone first, then unfreeze last N layers
FINETUNE_AFTER_EPOCH = int(EPOCHS * 0.6)  # epoch index to start finetuning (0-based)
UNFREEZE_LAST_N = 30                      # last N layers of backbone to unfreeze during fine-tune

# ---------- HELPERS ----------
def make_fixed_clip_paths(split="train", max_classes=MAX_CLASSES):
    """
    Scan directories and return two lists:
      list_of_clips: list of arrays of length FRAMES_PER_CLIP (strings: full frame paths)
      labels: corresponding integer labels
    Clips are padded by repeating last frame if there are fewer frames.
    """
    split_dir = os.path.join(SEQUENCES_DIR, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    class_names = sorted([d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d))])[:max_classes]
    class_map = {c: i for i, c in enumerate(class_names)}

    clip_paths = []
    labels = []

    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        for vid in vids:
            vid_dir = os.path.join(cls_dir, vid)
            frames = sorted([os.path.join(vid_dir, f) for f in os.listdir(vid_dir)
                             if f.lower().endswith(".jpg")])
            if not frames:
                continue
            # Build a fixed-length list of FRAMES_PER_CLIP paths, padding by repeating last frame
            if len(frames) >= FRAMES_PER_CLIP:
                chosen = frames[:FRAMES_PER_CLIP]
            else:
                chosen = frames + [frames[-1]] * (FRAMES_PER_CLIP - len(frames))
            clip_paths.append(chosen)
            labels.append(class_map[cls])

    return np.array(clip_paths, dtype=object), np.array(labels, dtype=np.int32), class_names

def _load_and_preprocess(paths):
    """
    Graph-mode loader: `paths` is a tensor of shape (FRAMES_PER_CLIP,) dtype string.
    Returns: clip tensor (FRAMES_PER_CLIP, H, W, 3) float32 normalized [0,1].
    """
    def _read_one(p):
        img = tf.io.read_file(p)
        img = tf.io.decode_jpeg(img, channels=3)  # handles jpg
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    clip = tf.map_fn(_read_one, paths, dtype=tf.float32)
    # clip shape: (FRAMES_PER_CLIP, H, W, 3)
    return clip

def augment_clip(clip):
    # clip shape: (FRAMES_PER_CLIP, H, W, 3)
    # simple augmentation applied frame-wise (consistent across frames)
    def aug_image(img):
        img = tf.image.random_flip_left_right(img)             # horizontal flip
        img = tf.image.random_brightness(img, max_delta=0.12)  # small brightness change
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return tf.clip_by_value(img, 0.0, 1.0)

    # Apply same augmentation independently per frame is ok; sign motion still preserved.
    clip = tf.map_fn(aug_image, clip, dtype=tf.float32)
    return clip

# ---------- DATASET CREATION ----------
def create_datasets():
    print("[INFO] Scanning train split...")
    train_paths, train_labels, class_names = make_fixed_clip_paths("train", max_classes=MAX_CLASSES)
    print(f"[INFO] Found {len(train_paths)} train clips across {len(class_names)} classes (capped).")

    print("[INFO] Scanning val split...")
    val_paths, val_labels, _ = make_fixed_clip_paths("val", max_classes=MAX_CLASSES)
    print(f"[INFO] Found {len(val_paths)} val clips across {len(class_names)} classes.")

    # Save class names for webcam / later
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    num_classes = len(class_names)

    # Build tf.data.Datasets (from_tensor_slices using object arrays works if dtype=object and strings)
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    # Map to load images in graph-mode
    def map_load(path_arr, lbl):
        clip = _load_and_preprocess(path_arr)
        clip = augment_clip(clip)  # train augmentation
        lbl_onehot = tf.one_hot(lbl, depth=num_classes)
        return clip, lbl_onehot

    def map_load_noaug(path_arr, lbl):
        clip = _load_and_preprocess(path_arr)
        lbl_onehot = tf.one_hot(lbl, depth=num_classes)
        return clip, lbl_onehot

    # Prepare train pipeline
    train_ds = (train_ds
                .shuffle(2048)
                .map(map_load, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .cache()                       # caches after map -> faster if fits RAM
                .prefetch(tf.data.AUTOTUNE)
                .repeat())                     # repeat indefinitely

    # Val pipeline (no augmentation, no repeat)
    val_ds = (val_ds
              .map(map_load_noaug, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE)
              .cache()
              .prefetch(tf.data.AUTOTUNE))

    steps_per_epoch = max(1, math.ceil(len(train_paths) / BATCH_SIZE))
    validation_steps = max(1, math.ceil(len(val_paths) / BATCH_SIZE))

    return train_ds, val_ds, steps_per_epoch, validation_steps, class_names

# ---------- MODEL ----------
def build_model(num_classes):
    input_shape = (FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3)
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # MobileNetV2 backbone applied TimeDistributed
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    backbone.trainable = False  # freeze initially

    x = layers.TimeDistributed(backbone)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)  # -> (batch, frames, feat)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model, backbone

# ---------- TRAINING ----------
def run_training():
    train_ds, val_ds, steps_per_epoch, validation_steps, class_names = create_datasets()
    num_classes = len(class_names)

    model, backbone = build_model(num_classes)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    es = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, restore_best_weights=True)
    ck = callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
    rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE_RLR, min_lr=1e-6)

    # Phase 1: train with backbone frozen
    initial_epochs = FINETUNE_AFTER_EPOCH
    if initial_epochs > 0:
        print(f"[INFO] Phase 1: training {initial_epochs} epochs with backbone frozen.")
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=initial_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[es, ck, rl],
            verbose=1
        )
    else:
        history1 = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    # Phase 2: unfreeze last N layers of backbone and fine-tune
    print(f"[INFO] Phase 2: unfreezing last {UNFREEZE_LAST_N} layers of backbone and fine-tuning remaining epochs.")
    # Find layer list in backbone, unfreeze last UNFREEZE_LAST_N layers
    if UNFREEZE_LAST_N > 0:
        for layer in backbone.layers[-UNFREEZE_LAST_N:]:
            layer.trainable = True
    backbone.trainable = True  # ensure backbone trainable (some layers may still be False)

    # Recompile with lower LR for fine-tuning
    model.compile(optimizer=optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    remaining_epochs = max(0, EPOCHS - initial_epochs)
    if remaining_epochs > 0:
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=initial_epochs + remaining_epochs,
            initial_epoch=initial_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[es, ck, rl],
            verbose=1
        )
    else:
        history2 = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    # Merge history dictionaries
    hist = {}
    for k in set(list(history1.get('history', {})) | set(list(history2.get('history', {})))):
        hist[k] = []
        if 'history' in history1:
            hist[k].extend(history1['history'].get(k, []))
        else:
            hist[k].extend(history1.get(k, []))
        if 'history' in history2:
            hist[k].extend(history2['history'].get(k, []))
        else:
            hist[k].extend(history2.get(k, []))

    # Save history & final model (ModelCheckpoint already saved best)
    with open(HISTORY_PATH, "w") as f:
        json.dump(hist, f)
    print(f"[INFO] Training history saved to {HISTORY_PATH}")

    # Save final model as well (optional)
    model.save(MODEL_PATH)
    print(f"[INFO] Final model saved to {MODEL_PATH}")

    # Plot training graphs
    if 'accuracy' in hist:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(hist['accuracy'], label='train acc')
        plt.plot(hist['val_accuracy'], label='val acc')
        plt.title("Accuracy")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(hist['loss'], label='train loss')
        plt.plot(hist['val_loss'], label='val loss')
        plt.title("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_graphs_v4.png")
        print("[INFO] Training graphs saved to training_graphs_v4.png")

if __name__ == "__main__":
    run_training()
