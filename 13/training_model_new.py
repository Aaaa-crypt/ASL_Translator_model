# train_prototypes.py
import os
import json
import math
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- CONFIG (edit paths & hyperparams) ----------------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 32                 # prototypes training benefits from moderately large batches
EMBED_DIM = 256                 # embedding size
EPOCHS = 30
MAX_CLASSES = 200               # limit for quick experiments; set to 2730 for full run
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\11_simple_model\prototypes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_WEIGHTS = os.path.join(OUTPUT_DIR, "best_prototype_weights.h5")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history_prototypes.json")
TEST_BAR_PLOT = os.path.join(OUTPUT_DIR, "test_bar_graph.png")

# ---------------- HELPERS: data listing & clip reading ----------------
def list_class_names(split="train"):
    split_dir = os.path.join(SEQUENCES_DIR, split)
    names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    if MAX_CLASSES:
        names = names[:MAX_CLASSES]
    return names

def get_clip_paths_for_split(split="train"):
    split_dir = os.path.join(SEQUENCES_DIR, split)
    class_names = list_class_names(split)
    class_map = {c:i for i,c in enumerate(class_names)}
    clip_list = []
    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        vids = sorted([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        for vid in vids:
            full = os.path.join(cls_dir, vid)
            frames = sorted([os.path.join(full, f) for f in os.listdir(full) if f.lower().endswith(".jpg")])
            if frames:
                clip_list.append((frames, class_map[cls]))
    return clip_list, class_names

def load_clip_numpy(frame_paths):
    # Load FRAMES_PER_CLIP frames, pad by repeating last if shorter
    frames = sorted(frame_paths)[:FRAMES_PER_CLIP]
    if len(frames) < FRAMES_PER_CLIP:
        # If clip shorter: replicate last frame to reach required length
        if len(frames) == 0:
            return None
        last = frames[-1]
        frames = frames + [last] * (FRAMES_PER_CLIP - len(frames))
    clip = []
    for p in frames[:FRAMES_PER_CLIP]:
        try:
            img = Image.open(p).convert("RGB").resize(IMG_SIZE)
            arr = np.array(img).astype("float32")
            arr = preprocess_input(arr)  # MobileNetV2 preprocessing
            clip.append(arr)
        except Exception:
            return None
    return np.stack(clip)  # (T, H, W, 3)

# ---------------- Create tf.data datasets (no one-hot) ----------------
def dataset_from_clip_list(clip_list, shuffle=False, repeat=False):
    def gen():
        for frames, lbl in clip_list:
            yield np.array(frames, dtype=object), int(lbl)
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    # map: load clips into numpy arrays via py_function
    def _map_py(frames, lbl):
        clip, label = tf.py_function(func=_py_load, inp=[frames, lbl], Tout=[tf.float32, tf.int32])
        clip.set_shape((FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3))
        label.set_shape(())
        return clip, label

    def _py_load(frames, lbl):
        # frames is a 1D tf.Tensor of bytes -> decode to Python list of paths
        paths = [p.decode("utf-8") for p in frames.numpy()]
        clip_np = load_clip_numpy(paths)
        if clip_np is None:
            # return zeros (will be filtered out later ideally)
            clip_np = np.zeros((FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        return clip_np.astype(np.float32), np.int32(lbl.numpy())

    ds = ds.map(_map_py, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    return ds

# ---------------- Build embedding model ----------------
def build_embedding_model(embed_dim=EMBED_DIM):
    # Input: (T,H,W,3)
    inp = layers.Input(shape=(FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3), name="frames")
    # backbone: MobileNetV2 applied per-frame
    backbone = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                                 input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # freeze initially (we'll keep it trainable in final training if desired)
    backbone.trainable = False
    x = layers.TimeDistributed(backbone)(inp)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)   # -> (B, T, C)
    # temporal aggregation: simple LSTM -> embedding
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dense(embed_dim)(x)
    x = layers.Lambda(lambda z: tf.math.l2_normalize(z, axis=1), name="l2_norm")(x)  # unit embeddings
    model = models.Model(inp, x, name="embed_net")
    return model

# ---------------- Prototypical losses & helpers ----------------
def prototypical_loss_and_acc(embeddings, labels):
    """
    embeddings: (B, D) normalized
    labels: (B,) int32
    We compute prototypes per-class from the batch and classify examples by negative squared distance.
    """
    labels = tf.cast(labels, tf.int32)
    unique_labels, idx = tf.unique(labels)  # unique classes in batch
    prototypes = []
    for l in tf.unstack(unique_labels):
        mask = tf.equal(labels, l)
        emb_c = tf.boolean_mask(embeddings, mask)
        prototype = tf.reduce_mean(emb_c, axis=0)
        prototypes.append(prototype)
    prototypes = tf.stack(prototypes, axis=0)  # (K, D)
    # compute squared euclidean distances between embeddings and prototypes
    # embeddings (B, D), prototypes (K, D) -> dists (B, K)
    dists = tf.reduce_sum(tf.square(tf.expand_dims(embeddings, 1) - tf.expand_dims(prototypes, 0)), axis=2)
    # logits = -dists
    logits = -dists
    # map labels to prototype indices
    # build mapping from unique_labels value -> index in 0..K-1
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=unique_labels, values=tf.range(tf.shape(unique_labels)[0], dtype=tf.int32)),
        default_value=-1
    )
    proto_idx = table.lookup(labels)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=proto_idx, logits=logits))
    pred = tf.argmax(logits, axis=1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, proto_idx), tf.float32))
    return loss, acc

def compute_prototypes_whole_set(embedding_model, clip_list):
    """
    Compute prototype per class by averaging embeddings over all training samples for that class.
    clip_list: list of (frames, label) like earlier
    Returns dict: class_idx -> prototype vector (D,)
    """
    accum = {}
    counts = {}
    for frames, lbl in clip_list:
        clip = load_clip_numpy(frames)
        if clip is None:
            continue
        emb = embedding_model.predict(clip[None, ...], verbose=0)[0]  # (D,)
        if lbl not in accum:
            accum[lbl] = emb.copy()
            counts[lbl] = 1
        else:
            accum[lbl] += emb
            counts[lbl] += 1
    # average
    prototypes = {lbl: (accum[lbl] / counts[lbl]) for lbl in accum.keys()}
    return prototypes

def eval_by_prototypes(embedding_model, prototypes, clip_list):
    """Evaluate accuracy and average loss-like distance on a clip_list (val/test)"""
    if not prototypes:
        return 0.0, float("inf")
    y_true = []
    y_pred = []
    dists_sum = 0.0
    n = 0
    # build stacked prototype matrix and label list for fast compute
    proto_labels = sorted(list(prototypes.keys()))
    proto_mat = np.stack([prototypes[l] for l in proto_labels], axis=0)  # (K,D)
    for frames, lbl in clip_list:
        clip = load_clip_numpy(frames)
        if clip is None:
            continue
        emb = embedding_model.predict(clip[None, ...], verbose=0)[0]  # (D,)
        # compute squared distances
        dif = proto_mat - emb[None, :]
        dists = np.sum(np.square(dif), axis=1)
        best = np.argmin(dists)
        pred_lbl = proto_labels[best]
        y_true.append(lbl)
        y_pred.append(pred_lbl)
        dists_sum += dists[best]
        n += 1
    if n == 0:
        return 0.0, float("inf")
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    avg_dist = dists_sum / n
    return acc, avg_dist

# ---------------- Prepare data lists ----------------
print("[INFO] Listing clips...")
train_clips, class_names_train = get_clip_paths_for_split("train")
val_clips, _ = get_clip_paths_for_split("val")
test_clips, _ = get_clip_paths_for_split("test")
num_classes = len(class_names_train)
print(f"[INFO] classes used: {num_classes}; train clips: {len(train_clips)}; val clips: {len(val_clips)}; test clips: {len(test_clips)}")

# ---------------- Build models & optimizer ----------------
embedding_model = build_embedding_model(EMBED_DIM)
optimizer = optimizers.Adam(learning_rate=1e-4)

# ---------------- tf.data for training (repeatable) ----------------
train_ds = dataset_from_clip_list(train_clips, shuffle=True, repeat=True)
val_ds = dataset_from_clip_list(val_clips, shuffle=False, repeat=False)
test_ds = dataset_from_clip_list(test_clips, shuffle=False, repeat=False)

steps_per_epoch = max(1, math.ceil(len(train_clips) / BATCH_SIZE))
print(f"[INFO] steps_per_epoch = {steps_per_epoch}")

# ---------------- Training loop (custom) ----------------
best_val_acc = -1.0
history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_dist": []}

for epoch in range(1, EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
    epoch_loss = []
    epoch_acc = []
    it = iter(train_ds)
    t0 = time.time()
    for step in range(steps_per_epoch):
        batch = next(it)  # (clips, labels)
        X_batch, y_batch = batch
        # gradient step
        with tf.GradientTape() as tape:
            embeddings = embedding_model(X_batch, training=True)  # (B,D), already L2-normalized
            loss, acc = prototypical_loss_and_acc(embeddings, y_batch)
        grads = tape.gradient(loss, embedding_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, embedding_model.trainable_weights))
        epoch_loss.append(float(loss))
        epoch_acc.append(float(acc))
        if (step + 1) % 50 == 0 or (step + 1) == steps_per_epoch:
            print(f" step {step+1}/{steps_per_epoch} — loss {loss:.4f} acc {acc:.4f}")

    t1 = time.time()
    train_loss = float(np.mean(epoch_loss))
    train_acc = float(np.mean(epoch_acc))
    print(f"Epoch {epoch} done in {t1-t0:.1f}s — train_loss {train_loss:.4f} train_acc {train_acc:.4f}")
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    # Compute prototypes from whole train set and evaluate val
    print("[INFO] Computing prototypes from whole training set (may take a while)...")
    prototypes = compute_prototypes_whole_set(embedding_model, train_clips)
    print(f"[INFO] {len(prototypes)} prototypes computed (out of {num_classes} classes).")
    val_acc, val_avg_dist = eval_by_prototypes(embedding_model, prototypes, val_clips)
    print(f"[INFO] Validation — acc: {val_acc:.4f} avg_dist: {val_avg_dist:.4f}")
    history["val_acc"].append(float(val_acc))
    history["val_dist"].append(float(val_avg_dist))

    # Save best weights by val_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        embedding_model.save_weights(MODEL_WEIGHTS)
        print(f"[INFO] New best val acc {best_val_acc:.4f} — weights saved to {MODEL_WEIGHTS}")

    # save incremental history
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

# ---------------- After training: load best weights & test ----------------
if os.path.exists(MODEL_WEIGHTS):
    embedding_model.load_weights(MODEL_WEIGHTS)
    print(f"[INFO] Loaded best weights from {MODEL_WEIGHTS}")

print("[INFO] Recomputing prototypes from full train set for final test...")
prototypes = compute_prototypes_whole_set(embedding_model, train_clips)
test_acc, test_avg_dist = eval_by_prototypes(embedding_model, prototypes, test_clips)
print(f"[INFO] Final Test acc: {test_acc:.4f} avg_dist: {test_avg_dist:.4f}")

# save test metrics and plot bar
test_metrics = {"accuracy": float(test_acc), "avg_dist": float(test_avg_dist)}
with open(os.path.join(OUTPUT_DIR, "test_prototypes.json"), "w") as f:
    json.dump(test_metrics, f, indent=2)

plt.figure(figsize=(6,4))
plt.bar(["Accuracy", "AvgDist"], [test_acc, test_avg_dist], color="#1f77b4")
plt.title("Test Metrics (Prototypes)")
plt.ylabel("Value")
plt.ylim(0, max(test_acc, test_avg_dist) * 1.2 if max(test_acc, test_avg_dist) > 0 else 1.0)
plt.savefig(TEST_BAR_PLOT)
plt.show()
print(f"[INFO] Test bar plot saved to {TEST_BAR_PLOT}")

# final save of history
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=2)
print("✅ Done.")
