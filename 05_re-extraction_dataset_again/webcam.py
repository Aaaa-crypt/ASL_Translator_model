# webcam.py
import cv2
import numpy as np
import os
from collections import deque
from tensorflow.keras.models import load_model

# -------- CONFIG --------
MODEL_PATH = r"C:\Users\garga\Documents\Maturarbeit\8\asl_seq_model_smart_v3.keras"
SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
FRAMES_PER_CLIP = 12       # Must match training
IMG_SIZE = (96, 96)

# -------- LOAD MODEL --------
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

# -------- AUTO LOAD CLASS NAMES --------
train_dir = os.path.join(SEQUENCES_DIR, "train")
class_names = sorted([d for d in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)
print(f"[INFO] Loaded {num_classes} classes from train folder.")

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
