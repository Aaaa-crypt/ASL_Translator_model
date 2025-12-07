import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- Paths ---
DATA_DIR = "archive/asl_alphabet_train"

IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 32
EPOCHS = 5  # adjust as needed

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

print("Classes found:", train_data.class_indices)
print("Number of classes:", len(train_data.class_indices))

# --- Model ---
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

# --- Save model ---
model.save("asl_model.keras")

# --- Save class names ---
idx_to_class = {v: k for k, v in train_data.class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("Saved class names:", class_names)

# --- Save training metrics to JSON ---
training_metrics = history.history

with open("training_metrics.json", "w", encoding="utf-8") as f:
    json.dump(training_metrics, f, indent=2)

print("Saved training metrics → training_metrics.json")

# --- Plot accuracy ---
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy.png", dpi=150)
plt.close()

# --- Plot loss ---
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss.png", dpi=150)
plt.close()

print("Saved accuracy.png and loss.png")
print("Training complete!")


