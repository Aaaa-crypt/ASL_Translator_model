# webcam.py
import os
import time
import cv2
import json
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------- CONFIG --------
MODEL_PATH = "asl_seq_model_smart_v4.keras"
CLASS_NAMES_PATH = "class_names_v4.json"
FRAMES_PER_CLIP = 12
IMG_SIZE = (96, 96)
PRED_SMOOTH = 4   # average predictions across last N clips to stabilize output
SHOW_CONF_THRESHOLD = 0.0  # show even low confidence; set >0.2 to filter

# -------- LOAD MODEL & CLASSES --------
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)
print("[INFO] Loading class names...")
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)
print(f"[INFO] Loaded {len(class_names)} classes.")

# -------- WEBCAM & BUFFERS --------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

frame_buffer = deque(maxlen=FRAMES_PER_CLIP)
pred_buffer = deque(maxlen=PRED_SMOOTH)

print("[INFO] Webcam started. Focus the webcam window and press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for mirror-like behavior (optional)
        # frame = cv2.flip(frame, 1)

        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(frame_rgb, IMG_SIZE)
        small = small.astype("float32") / 255.0
        frame_buffer.append(small)

        pred_label = ""
        confidence = 0.0

        if len(frame_buffer) == FRAMES_PER_CLIP:
            clip = np.expand_dims(np.array(frame_buffer), axis=0)  # (1, F, H, W, 3)
            preds = model.predict(clip, verbose=0)[0]             # (num_classes,)
            pred_buffer.append(preds)
            avg_pred = np.mean(np.stack(pred_buffer, axis=0), axis=0)
            idx = int(np.argmax(avg_pred))
            confidence = float(avg_pred[idx])
            if confidence >= SHOW_CONF_THRESHOLD:
                pred_label = class_names[idx]
            else:
                pred_label = ""

        # Display prediction and confidence
        overlay = frame.copy()
        if pred_label:
            text = f"{pred_label} ({confidence:.2f})"
            cv2.rectangle(overlay, (5,5), (430,60), (0,0,0), -1)  # semi background
            cv2.putText(overlay, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        # Show image
        cv2.imshow("ASL Live Prediction - press 'q' to quit", overlay)

        # Important: make the window active and press 'q' while it's focused
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stopped.")
