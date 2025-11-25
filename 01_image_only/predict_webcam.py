import cv2
import numpy as np
import tensorflow as tf
import os
import time

# Load trained model
model = tf.keras.models.load_model("asl_model.h5")  # or .keras if you used that format

# Auto-load class names from training folder
labels = sorted(os.listdir("archive/asl_alphabet_train"))  # Must match your training labels

# Image dimensions
img_size = 64

# Start webcam
cap = cv2.VideoCapture(0)

# Text buffer to display accumulated predictions
output_text = ""
last_pred = ""
last_time = time.time()

# To avoid duplicates, only add if 1s has passed since last update
prediction_delay = 1.0

print("📸 Live ASL prediction started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    roi = frame[100:400, 100:400]
    img = cv2.resize(roi, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img, verbose=0)
    class_index = np.argmax(predictions)
    predicted_label = labels[class_index]
    confidence = predictions[0][class_index]

    # Add to sentence every 1 second (avoid rapid repeats)
    if predicted_label != last_pred and (time.time() - last_time) > prediction_delay:
        if predicted_label == "space":
            output_text += " "
        elif predicted_label == "del":
            output_text = output_text[:-1]
        elif predicted_label == "nothing":
            pass  # Do nothing
        else:
            output_text += predicted_label
        last_pred = predicted_label
        last_time = time.time()

    # Display prediction box
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
    cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display output text buffer on right side of screen
    y0 = 50
    for i, line in enumerate([output_text]):
        y = y0 + i * 40
        cv2.putText(frame, f"Text: {line}", (450, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("ASL Live Prediction", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
