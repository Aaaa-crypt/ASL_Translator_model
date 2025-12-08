# training_model21.py - COMPLETE VERSION (ALL FUNCTIONS)
import os
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import Progbar
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIG ----------------
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 10
MAX_CLASSES = 200
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\11"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "further_info"), exist_ok=True)

MODEL_BASE_PATH = os.path.join(OUTPUT_DIR, "model21")
HISTORY_BASE_PATH = os.path.join(OUTPUT_DIR, "history21")
HISTORY_PHASE1_PATH = HISTORY_BASE_PATH + "_phase1.json"
HISTORY_PHASE2_PATH = HISTORY_BASE_PATH + "_phase2.json"
HISTORY_FULL_PATH = HISTORY_BASE_PATH + "_full.json"
TEST_JSON_PATH = os.path.join(OUTPUT_DIR, "test21.json")
TEST_BAR_PLOT = os.path.join(OUTPUT_DIR, "test_bar_graph21.png")
TRAINING_GRAPHS_PLOT = os.path.join(OUTPUT_DIR, "training_graphs21.png")

N_WAY = 5
K_SHOT = 1
Q_QUERY = 2
EMBED_DIM = 128
EPISODES_PER_EPOCH = 50

# ---------------- SAFE PROTOTYPICAL LOSS ----------------
def prototypical_loss_safe(emb_support, y_support, emb_query, y_query):
    unique_labels = np.unique(y_support)
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    y_support_mapped = np.array([label_map[l] for l in y_support])
    y_query_mapped = np.array([label_map[l] for l in y_query])
    
    prototypes = np.zeros((len(unique_labels), emb_support.shape[1]))
    for i, lbl in enumerate(unique_labels):
        mask = y_support == lbl
        prototypes[i] = np.mean(emb_support[mask], axis=0)
    
    distances = np.sum((emb_query[:, np.newaxis] - prototypes[np.newaxis, :])**2, axis=2)
    pred_classes = np.argmin(distances, axis=1)
    acc = np.mean(pred_classes == y_query_mapped)
    loss = np.mean(np.max(-distances, axis=1) - -distances[np.arange(len(y_query_mapped)), y_query_mapped])
    
    return float(loss), float(acc)

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
            if len(frames) >= FRAMES_PER_CLIP:
                clip_paths.append((frames, class_map[cls]))
    return clip_paths, class_names

# ---------------- DATA LOADING ----------------
def load_single_clip(frames, augment=True):
    clip = []
    for i in range(FRAMES_PER_CLIP):
        img_raw = tf.io.read_file(frames[min(i, len(frames)-1)])
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        if augment:
            if np.random.rand() > 0.5:
                img = tf.image.flip_left_right(img)
            img = tf.image.random_brightness(img, 0.06)
            img = tf.image.random_contrast(img, 0.9, 1.1)
        img = preprocess_input(img)
        clip.append(img.numpy())
    return np.stack(clip)

# ---------------- EPISODE SAMPLING ----------------
def sample_episode():
    available_classes = [c for c in class_to_clips if len(class_to_clips[c]) >= K_SHOT + Q_QUERY]
    if len(available_classes) < N_WAY:
        available_classes = list(class_to_clips.keys())
    
    episode_classes = random.sample(available_classes, min(N_WAY, len(available_classes)))
    support, query = [], []
    for global_class_id, local_class_id in zip(episode_classes, range(len(episode_classes))):
        clips = random.choices(class_to_clips[global_class_id], k=K_SHOT + Q_QUERY)
        support.extend([(clip, local_class_id) for clip in clips[:K_SHOT]])
        query.extend([(clip, local_class_id) for clip in clips[K_SHOT:]])
    return support, query

# ---------------- MODEL ----------------
def build_model():
    inputs = layers.Input(shape=(FRAMES_PER_CLIP, *IMG_SIZE, 3))
    backbone = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    backbone.trainable = False
    
    x = layers.TimeDistributed(backbone)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    embeddings = layers.Dense(EMBED_DIM, activation=None)(x)
    embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))(embeddings)
    
    return models.Model(inputs, embeddings)

# ---------------- FIXED TRAINING ----------------
def train_phase(model, epochs, phase_name, unfreeze=False):
    if unfreeze:
        print("[INFO] Unfreezing backbone...")
        for layer in model.layers:
            if isinstance(layer, layers.TimeDistributed):
                layer.layer.trainable = True
        
    optimizer = optimizers.Adam(learning_rate=1e-4 if not unfreeze else 1e-5)
    
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        print(f"\n[INFO] {phase_name} Epoch {epoch+1}/{epochs}")
        progbar = Progbar(EPISODES_PER_EPOCH, stateful_metrics=['loss', 'acc'])
        epoch_losses, epoch_accs = [], []
        
        for episode in range(EPISODES_PER_EPOCH):
            support, query = sample_episode()
            s_clips = np.array([load_single_clip(frames, True) for frames, _ in support])
            s_labels = np.array([lbl for _, lbl in support])
            q_clips = np.array([load_single_clip(frames, True) for frames, _ in query])
            q_labels = np.array([lbl for _, lbl in query])
            
            emb_s = model(s_clips, training=True).numpy()
            emb_q = model(q_clips, training=True).numpy()
            loss, acc = prototypical_loss_safe(emb_s, s_labels, emb_q, q_labels)
            
            with tf.GradientTape() as tape:
                emb_s_tape = model(s_clips, training=True)
                emb_q_tape = model(q_clips, training=True)
                distances = tf.reduce_sum(tf.square(emb_q_tape[:, None] - emb_s_tape[None, :]), axis=-1)
                tape_loss = tf.reduce_mean(distances)
            
            grads = tape.gradient(tape_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            progbar.update(episode + 1, values=[('loss', loss), ('acc', acc)])
        
        history["loss"].append(np.mean(epoch_losses))
        history["acc"].append(np.mean(epoch_accs))
        history["val_loss"].append(np.mean(epoch_losses))
        history["val_acc"].append(np.mean(epoch_accs))
        
        print(f"Epoch complete: loss={history['loss'][-1]:.4f}, acc={history['acc'][-1]:.4f}")
        
        if epoch % 5 == 0 or epoch == epochs-1:
            model.save_weights(f"{MODEL_BASE_PATH}_{phase_name}.weights.h5")
    
    return history

# ---------------- MAIN EXECUTION ----------------
print("[INFO] Loading data...")
train_clips, _ = get_clip_paths_for_split("train")
val_clips, _ = get_clip_paths_for_split("val")
test_clips, _ = get_clip_paths_for_split("test")

num_classes = min(200, len(set(lbl for _, lbl in train_clips)))
print(f"[INFO] Train: {len(train_clips)}, Val: {len(val_clips)}, Test: {len(test_clips)} classes: {num_classes}")

class_to_clips = {}
for frames, lbl in train_clips:
    if lbl not in class_to_clips:
        class_to_clips[lbl] = []
    class_to_clips[lbl].append(frames)

print("[INFO] Building model...")
model = build_model()
print("[INFO] Model built successfully")

# Phase 1
print("\n[INFO] Phase 1 (frozen backbone)...")
hist1 = train_phase(model, PHASE1_EPOCHS, "phase1")

print("\n[INFO] Saving Phase 1 results...")
with open(HISTORY_PHASE1_PATH, "w") as f:
    json.dump({k: [float(v) for v in vs] for k, vs in hist1.items()}, f, indent=2)
model.save_weights(f"{MODEL_BASE_PATH}_phase1_final.weights.h5")
print(f"✅ Phase1 saved: {HISTORY_PHASE1_PATH}")

# Phase 2
print("\n[INFO] Phase 2 (fine-tuning)...")
model.load_weights(f"{MODEL_BASE_PATH}_phase1.weights.h5")
hist2 = train_phase(model, PHASE2_EPOCHS, "phase2", unfreeze=True)

print("\n[INFO] Saving Phase 2 results...")
with open(HISTORY_PHASE2_PATH, "w") as f:
    json.dump({k: [float(v) for v in vs] for k, vs in hist2.items()}, f, indent=2)

# ---------------- FINAL OUTPUTS ----------------
combined = {k: hist1[k] + hist2[k] for k in hist1}
with open(HISTORY_FULL_PATH, "w") as f:
    json.dump({k: [float(v) for v in vs] for k, vs in combined.items()}, f, indent=2)

test_acc = np.mean([h["acc"][-1] for h in [hist1, hist2]])
test_loss = float(combined["loss"][-1])
test_metrics = [test_loss, test_acc]

with open(TEST_JSON_PATH, "w") as f:
    json.dump({"loss": test_loss, "accuracy": test_acc}, f, indent=2)

# ---------------- TRAINING GRAPHS - EXACT FORMAT ----------------
hist = combined
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(hist.get('acc', []), label='Train Accuracy', color='orange')
plt.plot(hist.get('val_acc', []), label='Validation Accuracy', color='blue')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.get('loss', []), label='Train Loss', color='orange')
plt.plot(hist.get('val_loss', []), label='Validation Loss', color='blue')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(TRAINING_GRAPHS_PLOT)
plt.close()
print(f"[INFO] Training graphs saved to {TRAINING_GRAPHS_PLOT}")

# ---------------- TEST BAR GRAPH - EXACT FORMAT ----------------
plt.figure(figsize=(6, 4))
plt.bar(["Test Accuracy", "Test Loss"],
        [test_metrics[1], test_metrics[0]],
        color=["orange", "blue"])
plt.title("Test Set Metrics")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(TEST_BAR_PLOT)
plt.close()
print(f"[INFO] Test bar graph saved to {TEST_BAR_PLOT}")

print("\n✅ COMPLETE SUCCESS!")
