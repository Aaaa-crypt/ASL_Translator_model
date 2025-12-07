# training_model21.py
import os
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIG ----------------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 16  # used only for val/test
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 40
MAX_CLASSES = 200
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\11"
FURTHER_INFO_DIR = os.path.join(OUTPUT_DIR, "further_info")
os.makedirs(FURTHER_INFO_DIR, exist_ok=True)

MODEL_BASE_PATH = os.path.join(OUTPUT_DIR, "model21")
HISTORY_BASE_PATH = os.path.join(OUTPUT_DIR, "history21")
HISTORY_PHASE1_PATH = HISTORY_BASE_PATH + "_phase1.json"
HISTORY_PHASE2_PATH = HISTORY_BASE_PATH + "_phase2.json"
HISTORY_FULL_PATH = HISTORY_BASE_PATH + "_full.json"
TEST_JSON_PATH = os.path.join(OUTPUT_DIR, "test21.json")
TEST_BAR_PLOT = os.path.join(OUTPUT_DIR, "test_bar_graph21.png")

# Few-shot episode config
N_WAY = 10      # number of classes per episode
K_SHOT = 2      # support examples per class
Q_QUERY = 4     # query examples per class
EMBED_DIM = 256

# ---------------- HELPERS: CLIP LISTS ----------------
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
            frames = sorted(
                [os.path.join(full_vid_dir, f) for f in os.listdir(full_vid_dir)
                 if f.lower().endswith(".jpg")]
            )
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

# ---------------- IMAGE / CLIP LOADING ----------------
def random_zoom_image(img, zoom_range=(0.96, 1.04)):
    h, w = IMG_SIZE
    zoom_factor = tf.random.uniform([], zoom_range[0], zoom_range[1])
    new_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)
    img = tf.image.resize(img, (new_h, new_w))
    img = tf.image.resize_with_crop_or_pad(img, h, w)
    return img

def load_clip_py(frame_paths, augment=True):
    frame_paths = [p.decode("utf-8") for p in frame_paths.numpy()]
    clip = []
    for i in range(FRAMES_PER_CLIP):
        idx = min(i, len(frame_paths) - 1)
        img_raw = tf.io.read_file(frame_paths[idx])
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.06)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            img = random_zoom_image(img)
        img = preprocess_input(img)
        clip.append(img)
    clip = tf.stack(clip)
    return clip

def tf_load_clip(frame_paths, label, augment=True):
    clip = tf.py_function(
        func=lambda fp: load_clip_py(fp, augment=augment),
        inp=[frame_paths],
        Tout=tf.float32
    )
    clip.set_shape([FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3])
    return clip, label

def create_tf_dataset_from_list(clip_list, repeat=False, augment=False):
    ds = tf.data.Dataset.from_generator(
        lambda: ((np.array(frames, dtype=object), int(lbl)) for frames, lbl in clip_list),
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    ds = ds.map(lambda fp, lbl: tf_load_clip(fp, lbl, augment=augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.shuffle(buffer_size=max(1024, len(clip_list))).repeat()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------- EPISODE SAMPLING (FEW-SHOT) ----------------
# Build index: class_id -> list of clip indices
class_to_indices = {cid: [] for cid in range(num_classes)}
for idx, (_, lbl) in enumerate(train_clips):
    class_to_indices[lbl].append(idx)

def sample_episode(n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY):
    """Return support and query lists of (frame_paths, class_id)."""
    # pick classes that have at least k_shot + q_query samples if possible
    eligible_classes = [c for c, idxs in class_to_indices.items()
                        if len(idxs) >= k_shot + q_query]
    if len(eligible_classes) < n_way:
        eligible_classes = [c for c in class_to_indices.keys() if len(class_to_indices[c]) > 0]
    chosen_classes = random.sample(eligible_classes, n_way)

    support = []
    query = []
    for epi_cls in chosen_classes:
        idxs = class_to_indices[epi_cls]
        if len(idxs) >= k_shot + q_query:
            chosen = random.sample(idxs, k_shot + q_query)
        else:
            chosen = random.choices(idxs, k=k_shot + q_query)
        support_ids = chosen[:k_shot]
        query_ids = chosen[k_shot:]
        for i in support_ids:
            support.append(train_clips[i])
        for i in query_ids:
            query.append(train_clips[i])

    return support, query, chosen_classes

def episode_to_batch(support, query):
    """Convert support/query lists to tensors: (clips, labels)."""
    # support
    s_frames = []
    s_labels = []
    for frame_paths, lbl in support:
        frames = np.array(frame_paths, dtype=object)
        s_frames.append(frames)
        s_labels.append(lbl)
    # query
    q_frames = []
    q_labels = []
    for frame_paths, lbl in query:
        frames = np.array(frame_paths, dtype=object)
        q_frames.append(frames)
        q_labels.append(lbl)
    return (s_frames, np.array(s_labels, dtype=np.int32)), \
           (q_frames, np.array(q_labels, dtype=np.int32))

def load_batch_clips(frame_paths_array, labels_array, augment=True):
    """Load a batch of clips (Python side, then stacked to tensor)."""
    clips = []
    for frame_paths in frame_paths_array:
        clip = load_clip_py(tf.constant(frame_paths, dtype=tf.string), augment=augment)
        clips.append(clip.numpy())
    clips = np.stack(clips, axis=0)  # (B, T, H, W, C)
    return clips, labels_array

# ---------------- MODEL: EMBEDDING NETWORK ----------------
def temporal_conv_block(x, filters=128, kernel_size=3):
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x

class TemporalAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units, activation="tanh")
        self.V = layers.Dense(1)

    def call(self, inputs):
        score = self.V(self.W(inputs))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

def build_embedding_model(backbone_trainable=True, lr=1e-4, embed_dim=EMBED_DIM):
    frames_input = layers.Input(
        shape=(FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3), name="frames"
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = backbone_trainable

    x = layers.TimeDistributed(backbone)(frames_input)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = temporal_conv_block(x, filters=256, kernel_size=3)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = TemporalAttention(128)(x)
    x = layers.Dropout(0.5)(x)

    # Embedding head
    emb = layers.Dense(embed_dim, activation=None, name="embedding")(x)
    emb = tf.nn.l2_normalize(emb, axis=-1)  # L2-normalized embedding

    model = models.Model(frames_input, emb)
    optimizer = optimizers.Adam(learning_rate=lr)
    return model, optimizer

# ---------------- PROTOTYPICAL LOSS ----------------
def prototypical_loss(emb_support, y_support, emb_query, y_query):
    """
    emb_support: (S, D)
    y_support: (S,)
    emb_query: (Q, D)
    y_query: (Q,)
    """
    y_support = tf.cast(y_support, tf.int32)
    y_query = tf.cast(y_query, tf.int32)
    classes = tf.unique(y_support)[0]            # (N_way,)
    n_classes = tf.shape(classes)[0]

    # compute prototypes (class means)
    prototypes = []
    for c in tf.unstack(classes):
        mask = tf.equal(y_support, c)
        class_emb = tf.boolean_mask(emb_support, mask)
        proto = tf.reduce_mean(class_emb, axis=0)
        prototypes.append(proto)
    prototypes = tf.stack(prototypes, axis=0)    # (N_way, D)

    # distances from query to prototypes
    # expand: query (Q,1,D), prototypes (1,N,D)
    q_exp = tf.expand_dims(emb_query, axis=1)
    p_exp = tf.expand_dims(prototypes, axis=0)
    distances = tf.reduce_sum((q_exp - p_exp) ** 2, axis=-1)  # (Q, N_way)

    # convert y_query to indices 0..N_way-1
    # map global class_id -> [0..N_way)
    class_to_epi = {int(c.numpy()): i for i, c in enumerate(classes)}
    y_query_epi = tf.constant(
        [class_to_epi[int(c.numpy())] for c in y_query], dtype=tf.int32
    )

    log_p_y = tf.nn.log_softmax(-distances, axis=-1)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_query_epi, logits=-distances
    )
    loss = tf.reduce_mean(loss)

    # accuracy
    preds = tf.argmax(-distances, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_query_epi), tf.float32))

    return loss, acc

# ---------------- TRAINING LOOP (EPISODIC) ----------------
def run_phase(model, optimizer, phase_name, epochs, backbone_unfreeze=False):
    train_history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    if backbone_unfreeze:
        print("[INFO] Unfreezing backbone for fine-tuning...")
        for layer in model.layers:
            if isinstance(layer, layers.TimeDistributed):
                if hasattr(layer.layer, "trainable"):
                    layer.layer.trainable = True

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )

    # Build simple val dataset (classification with nearest prototype)
    val_ds = create_tf_dataset_from_list(val_clips, repeat=False, augment=False)

    ckpt_path = f"{MODEL_BASE_PATH}_{phase_name}.weights.h5"
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path + "_dir", max_to_keep=1)

    for epoch in range(epochs):
        print(f"\n[INFO] {phase_name} - Epoch {epoch+1}/{epochs}")
        # --- TRAIN EPISODES ---
        num_episodes = max(1, len(train_clips) // (N_WAY * (K_SHOT + Q_QUERY)))
        epoch_losses = []
        epoch_accs = []

        for _ in range(num_episodes):
            support, query, _ = sample_episode(N_WAY, K_SHOT, Q_QUERY)
            (s_frames, s_labels), (q_frames, q_labels) = episode_to_batch(
                support, query
            )

            s_clips, s_labels = load_batch_clips(s_frames, s_labels, augment=True)
            q_clips, q_labels = load_batch_clips(q_frames, q_labels, augment=True)

            with tf.GradientTape() as tape:
                emb_s = model(s_clips, training=True)
                emb_q = model(q_clips, training=True)
                loss, acc = prototypical_loss(
                    emb_s, s_labels, emb_q, q_labels
                )

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_losses.append(float(loss.numpy()))
            epoch_accs.append(float(acc.numpy()))

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(np.mean(epoch_accs))
        train_history["loss"].append(train_loss)
        train_history["acc"].append(train_acc)

        # --- VALIDATION: classify with prototypes built per-batch ---
        val_losses = []
        val_accs = []
        for x_batch, y_batch in val_ds:
            # For validation, build "support" from the same batch (K=N/2) for each class in the batch
            x_np = x_batch.numpy()
            y_np = y_batch.numpy()
            emb = model(x_np, training=False).numpy()

            # group by label and split into pseudo support / query
            labels_unique = np.unique(y_np)
            s_emb_list = []
            s_lab_list = []
            q_emb_list = []
            q_lab_list = []
            for lbl in labels_unique:
                idxs = np.where(y_np == lbl)[0]
                if len(idxs) < 2:
                    continue
                split = len(idxs) // 2
                s_idx = idxs[:split]
                q_idx = idxs[split:]
                s_emb_list.append(emb[s_idx])
                s_lab_list.append(np.full(len(s_idx), lbl, dtype=np.int32))
                q_emb_list.append(emb[q_idx])
                q_lab_list.append(np.full(len(q_idx), lbl, dtype=np.int32))

            if not s_emb_list:
                continue

            s_emb = np.concatenate(s_emb_list, axis=0)
            s_lab = np.concatenate(s_lab_list, axis=0)
            q_emb = np.concatenate(q_emb_list, axis=0)
            q_lab = np.concatenate(q_lab_list, axis=0)

            s_emb_tf = tf.convert_to_tensor(s_emb, dtype=tf.float32)
            q_emb_tf = tf.convert_to_tensor(q_emb, dtype=tf.float32)
            s_lab_tf = tf.convert_to_tensor(s_lab, dtype=tf.int32)
            q_lab_tf = tf.convert_to_tensor(q_lab, dtype=tf.int32)

            v_loss, v_acc = prototypical_loss(s_emb_tf, s_lab_tf, q_emb_tf, q_lab_tf)
            val_losses.append(float(v_loss.numpy()))
            val_accs.append(float(v_acc.numpy()))

        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0
        train_history["val_loss"].append(val_loss)
        train_history["val_acc"].append(val_acc)

        print(f"[INFO] {phase_name} Epoch {epoch+1}: "
              f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # callbacks: early stopping + lr scheduler + checkpoint
        lr_scheduler.on_epoch_end(epoch, logs={"val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_manager.save()
            print(f"[INFO] {phase_name}: Saved new best checkpoint.")
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"[INFO] {phase_name}: Early stopping.")
                break

    return train_history, ckpt_manager.latest_checkpoint

# ---------------- BUILD & TRAIN ----------------
print("[INFO] Building embedding model (phase1, frozen backbone)...")
model, optimizer = build_embedding_model(backbone_trainable=False, lr=1e-4)

hist1, best_ckpt_phase1 = run_phase(model, optimizer, "phase1", PHASE1_EPOCHS)

# Save phase1 history
try:
    with open(HISTORY_PHASE1_PATH, "w") as f:
        json.dump(hist1, f, indent=2)
    print(f"[INFO] Phase1 history saved to {HISTORY_PHASE1_PATH}")
except Exception as e:
    print(f"[WARN] Could not save Phase1 history: {e}")

# Phase 2: fine-tune backbone
print("[INFO] Fine-tuning backbone (phase2)...")
# Reload best phase1 checkpoint if available
if best_ckpt_phase1:
    print(f"[INFO] Restoring best Phase1 checkpoint from {best_ckpt_phase1}")
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt.restore(best_ckpt_phase1).expect_partial()

# new optimizer with lower LR
optimizer = optimizers.Adam(learning_rate=1e-5)
hist2, best_ckpt_phase2 = run_phase(model, optimizer, "phase2", PHASE2_EPOCHS, backbone_unfreeze=True)

# Save phase2 history
try:
    with open(HISTORY_PHASE2_PATH, "w") as f:
        json.dump(hist2, f, indent=2)
    print(f"[INFO] Phase2 history saved to {HISTORY_PHASE2_PATH}")
except Exception as e:
    print(f"[WARN] Could not save Phase2 history: {e}")

# Combine histories
combined_history = {}
for k in set(list(hist1.keys()) + list(hist2.keys())):
    combined_history[k] = hist1.get(k, []) + hist2.get(k, [])

try:
    with open(HISTORY_FULL_PATH, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"[INFO] Full combined history saved to {HISTORY_FULL_PATH}")
except Exception as e:
    print(f"[WARN] Could not save full history: {e}")

# ---------------- TEST EVALUATION ----------------
# Load best model from phase2 (or phase1 fallback)
best_ckpt = best_ckpt_phase2 or best_ckpt_phase1
if best_ckpt:
    print(f"[INFO] Restoring best checkpoint for testing from {best_ckpt}")
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt.restore(best_ckpt).expect_partial()
else:
    print("[WARN] No checkpoint found, using current model weights for test.")

test_ds = create_tf_dataset_from_list(test_clips, repeat=False, augment=False)

# For test, evaluate classification with batch-wise prototypes same as val
test_losses = []
test_accs = []
for x_batch, y_batch in test_ds:
    x_np = x_batch.numpy()
    y_np = y_batch.numpy()
    emb = model(x_np, training=False).numpy()

    labels_unique = np.unique(y_np)
    s_emb_list = []
    s_lab_list = []
    q_emb_list = []
    q_lab_list = []
    for lbl in labels_unique:
        idxs = np.where(y_np == lbl)[0]
        if len(idxs) < 2:
            continue
        split = len(idxs) // 2
        s_idx = idxs[:split]
        q_idx = idxs[split:]
        s_emb_list.append(emb[s_idx])
        s_lab_list.append(np.full(len(s_idx), lbl, dtype=np.int32))
        q_emb_list.append(emb[q_idx])
        q_lab_list.append(np.full(len(q_idx), lbl, dtype=np.int32))

    if not s_emb_list:
        continue

    s_emb = np.concatenate(s_emb_list, axis=0)
    s_lab = np.concatenate(s_lab_list, axis=0)
    q_emb = np.concatenate(q_emb_list, axis=0)
    q_lab = np.concatenate(q_lab_list, axis=0)

    s_emb_tf = tf.convert_to_tensor(s_emb, dtype=tf.float32)
    q_emb_tf = tf.convert_to_tensor(q_emb, dtype=tf.float32)
    s_lab_tf = tf.convert_to_tensor(s_lab, dtype=tf.int32)
    q_lab_tf = tf.convert_to_tensor(q_lab, dtype=tf.int32)

    t_loss, t_acc = prototypical_loss(s_emb_tf, s_lab_tf, q_emb_tf, q_lab_tf)
    test_losses.append(float(t_loss.numpy()))
    test_accs.append(float(t_acc.numpy()))

test_loss = float(np.mean(test_losses)) if test_losses else 0.0
test_acc = float(np.mean(test_accs)) if test_accs else 0.0

try:
    with open(TEST_JSON_PATH, "w") as f:
        json.dump({"loss": test_loss, "accuracy": test_acc}, f, indent=2)
    print(f"[INFO] Test metrics saved to {TEST_JSON_PATH}")
except Exception as e:
    print(f"[WARN] Could not save test metrics: {e}")

# ---------------- PLOTS ----------------
try:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(combined_history.get("acc", []), label="Train")
    plt.plot(combined_history.get("val_acc", []), label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(combined_history.get("loss", []), label="Train")
    plt.plot(combined_history.get("val_loss", []), label="Val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    training_graph_path = os.path.join(OUTPUT_DIR, "training_graphs21.png")
    plt.savefig(training_graph_path)
    print(f"[INFO] Training graphs saved to {training_graph_path}")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy", "Loss"], [test_acc, test_loss])
    plt.title("Test Metrics")
    plt.savefig(TEST_BAR_PLOT)
    print(f"[INFO] Test bar plot saved to {TEST_BAR_PLOT}")
    plt.close()
except Exception as e:
    print(f"[WARN] Could not create or save plots: {e}")

print("✅ Done.")
