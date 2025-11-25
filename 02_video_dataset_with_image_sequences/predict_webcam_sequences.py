# predict_webcam_sequences.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

MODEL_PATH = r"C:\Users\GG\Documents\Maturarbeit\asl_seq_model.h5"
IMG_SIZE = (112, 112)
FRAMES_PER_CLIP = 6

# Load model
model = load_model(MODEL_PATH)

# Initialize deque to store last FRAMES_PER_CLIP frames
frame_buffer = deque(maxlen=FRAMES_PER_CLIP)

# Load class names (from training)
import json
with open(r"C:\Users\GG\Documents\Maturarbeit\class_names.json", "r") as f:
    class_names = json.load(f)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_norm = frame_resized / 255.0
    frame_buffer.append(frame_norm)

    # Only predict when we have enough frames
    if len(frame_buffer) == FRAMES_PER_CLIP:
        clip_input = np.expand_dims(np.array(frame_buffer), axis=0)  # shape: (1, 6, 112, 112, 3)
        preds = model.predict(clip_input, verbose=0)
        label = class_names[np.argmax(preds)]

        # Display on webcam
        cv2.putText(frame, f"Prediction: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("ASL Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
