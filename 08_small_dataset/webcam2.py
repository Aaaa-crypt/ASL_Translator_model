# webcam2.py
import cv2
import numpy as np
import os
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers, models
import time

# -------- CONFIG --------
MODEL_PATH = r"C:\Users\garga\Documents\Maturarbeit\08_small_dataset\asl_seq_model_phase2.weights.h5"
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\asl_sequences_mixed"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
LATENCY = 2.5  # seconds before registering same sign again

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

recognized_words = []
last_prediction_time = 0
last_pred_label = ""

# Define window size
window_width = 1280
window_height = 480
half_width = window_width // 2

# Button coordinates
button_x1, button_y1, button_x2, button_y2 = 10, window_height - 60, 150, window_height - 10

# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global recognized_words
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x1 <= x <= button_x2 and button_y1 <= y <= button_y2:
            recognized_words = []

cv2.namedWindow("ASL Live Prediction")
cv2.setMouseCallback("ASL Live Prediction", mouse_callback)

print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for uniformity
    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_norm = frame_resized / 255.0
    frame_buffer.append(frame_norm)

    pred_label = ""
    confidence = 0.0

    # Predict only when buffer is full
    if len(frame_buffer) == FRAMES_PER_CLIP:
        clip = np.expand_dims(np.array(frame_buffer), axis=0)  # (1,12,96,96,3)
        preds = model.predict(clip, verbose=0)
        idx = np.argmax(preds[0])
        pred_label = class_names[idx]
        confidence = preds[0][idx]

        # Check latency to avoid double typing
        current_time = time.time()
        if pred_label != last_pred_label or (current_time - last_prediction_time) > LATENCY:
            recognized_words.append(pred_label)
            last_pred_label = pred_label
            last_prediction_time = current_time

    # -------- BUILD DISPLAY --------
    # Resize webcam feed to left half
    left_frame = cv2.resize(frame, (half_width, window_height))
    
    # Create white right half
    right_frame = np.ones((window_height, half_width, 3), dtype=np.uint8) * 255

    # Display recognized words
    y0, dy = 50, 40
    for i, word in enumerate(recognized_words[-10:]):  # show last 10 words
        y = y0 + i * dy
        cv2.putText(right_frame, word, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Draw Clear button
    cv2.rectangle(right_frame, (button_x1, button_y1), (button_x2, button_y2), (0, 0, 255), -1)
    cv2.putText(right_frame, "Clear", (button_x1 + 10, button_y2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Combine left and right halves
    combined_frame = np.hstack((left_frame, right_frame))

    # Optionally, overlay current prediction on webcam feed
    if pred_label:
        text = f"{pred_label} ({confidence:.2f})"
        cv2.putText(combined_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show combined frame
    cv2.imshow("ASL Live Prediction", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
