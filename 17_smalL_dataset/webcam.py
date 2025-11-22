# webcam.py
import cv2
import numpy as np
import os
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers, models

# -------- CONFIG --------
MODEL_PATH = r"C:\Users\garga\Documents\Maturarbeit\17_small_dataset\asl_seq_model_phase2.weights.h5"
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\asl_sequences_mixed"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)

# -------- LOAD CLASS NAMES --------
train_dir = os.path.join(SEQUENCES_DIR, "train")
class_names = sorted([d for d in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)
print(f"[INFO] Loaded {num_classes} classes from train folder.")

# -------- BUILD MODEL ARCHITECTURE --------
def temporal_conv_block(x, filters=128):
    x = layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
    return layers.BatchNormalization()(x)

class TemporalAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units, activation="tanh")
        self.V = layers.Dense(1)
    def call(self, x):
        w = tf.nn.softmax(self.V(self.W(x)), axis=1)
        return tf.reduce_sum(w * x, axis=1)

def build_model(num_classes):
    inp = layers.Input((FRAMES_PER_CLIP, IMG_SIZE[0], IMG_SIZE[1], 3))
    base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base.trainable = True
    x = layers.TimeDistributed(base)(inp)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = temporal_conv_block(x, 256)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = TemporalAttention(128)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    return model

# -------- LOAD MODEL --------
print("[INFO] Loading model architecture and weights...")
model = build_model(num_classes)
model.load_weights(MODEL_PATH)
print("[INFO] Model loaded successfully.")

# -------- SETUP WEBCAM --------
cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=FRAMES_PER_CLIP)
pred_label = ""
confidence = 0.0

print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_norm = frame_resized / 255.0
    frame_buffer.append(frame_norm)

    # Predict only when buffer is full
    if len(frame_buffer) == FRAMES_PER_CLIP:
        clip = np.expand_dims(np.array(frame_buffer), axis=0)  # (1,12,96,96,3)
        preds = model.predict(clip, verbose=0)
        idx = np.argmax(preds[0])
        pred_label = class_names[idx]
        confidence = preds[0][idx]

    # Display prediction
    if pred_label:
        text = f"{pred_label} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
