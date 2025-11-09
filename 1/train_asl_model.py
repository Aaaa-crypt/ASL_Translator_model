import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- Paths ---
# Use your project-relative path; this matches your actual folder layout
DATA_DIR = "archive/asl_alphabet_train"

IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 32
EPOCHS = 5  # adjust later if you like

# --- Data ---
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# (Optional) sanity check: you should see 29 classes (A-Z, del, nothing, space)
print("Classes found:", train_data.class_indices)
print("Number of classes:", len(train_data.class_indices))

# --- Model ---
# Use an explicit Input layer (avoids the Keras warning)
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Train ---
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# --- Save model in modern Keras format (recommended) ---
model.save("asl_model.keras")

# (Optionally also save HDF5 if you want)
# model.save("asl_model.h5")

# --- Save class names in the exact order the model uses ---
# Invert mapping index->class and write an ordered list by index
idx_to_class = {v: k for k, v in train_data.class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("Saved class names:", class_names)

# --- Plot & save accuracy and loss curves ---
# Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy.png", dpi=150)
plt.show()

# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss.png", dpi=150)
plt.show()
