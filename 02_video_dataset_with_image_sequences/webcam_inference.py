import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# -------- CONFIG --------
MODEL_PATH = "asl_seq_model.h5"
SEQUENCES_DIR = r"C:\Users\GG\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences"
FRAMES_PER_CLIP = 5   # Must match training setting
IMG_SIZE = (112, 112)

# -------- LOAD MODEL & CLASSES --------
print("[INFO] Loading trained model...")
model = load_model(MODEL_PATH)

# Load class names from training folder
train_dir = os.path.join(SEQUENCES_DIR, "train")
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"[INFO] Loaded {len(class_names)} classes.")

# -------- CAPTURE FROM WEBCAM --------
cap = cv2.VideoCapture(0)  # 0 = default webcam
frames_buffer = []
pred_label = ""
confidence = 0.0

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_norm = frame_resized / 255.0
    frames_buffer.append(frame_norm)

    # Once we have a full clip → make prediction
    if len(frames_buffer) == FRAMES_PER_CLIP:
        clip = np.array([frames_buffer])  # shape (1, FRAMES, H, W, 3)
        preds = model.predict(clip, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_label = class_names[pred_idx]
        confidence = preds[0][pred_idx]
        frames_buffer = []  # reset for next clip

    # ----- Draw prediction on screen -----
    if pred_label:
        display_text = f"{pred_label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 2

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(display_text, font, scale, thickness)

        # Position (bottom-left corner of text)
        x, y = 30, 50

        # Draw filled rectangle as background
        cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (255, 255, 255), -1)

        # Draw text on top
        cv2.putText(frame, display_text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Show live video with overlay
    cv2.imshow("ASL Translation", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
