import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

# Dataset Path
dataset_path = ""  # Change to the path of your dataset

# Image Preprocessing
image_size = (224, 224)  # MobileNetV2 requires 224x224 input size
batch_size = 32

# Data Augmentation and Splitting
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 20% for validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Load Pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze Base Model Layers
base_model.trainable = False

# Add Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce feature dimensions
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)  # Prevent overfitting
output = Dense(len(train_data.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # Increase this for better performance
)

# Save the Keras Model
model.save("fruit_ripeness_model_transfer.h5")
print("Model saved as fruit_ripeness_model_transfer.h5")

# Convert the Saved Model to TensorFlow Lite
def convert_to_tflite(model_path, tflite_output_path):
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model successfully converted to TensorFlow Lite and saved as {tflite_output_path}")

# Convert the trained model to TensorFlow Lite
convert_to_tflite("fruit_ripeness_model_transfer.h5", "fruit_ripeness_model.tflite")

# Plot Training and Validation Accuracy & Loss
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss")

plt.show()

# Test the TFLite Model
tflite_model_path = "fruit_ripeness_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to MobileNetV2 input size
    image = np.expand_dims(image / 255.0, axis=0).astype(np.float32)  # Normalize and convert to float32

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)

    return predicted_class

# Map class indices to labels
class_labels = train_data.class_indices
class_labels = {v: k for k, v in class_labels.items()}  # Reverse key-value pairs

# Test the TFLite Model
test_image_path = "path_to_test_image.jpg"  # Replace with a test image path
predicted_class = predict_tflite_image(test_image_path)
print(f"The predicted ripeness is: {class_labels[predicted_class]}")
