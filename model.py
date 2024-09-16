import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set up paths
base_dir = 'Dataset'  # This is the correct base directory
train_dir = os.path.join(base_dir, 'Train')
val_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

# Load MobileNetV2 as the base model
base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3),
                                               include_top=False,  # Exclude the top classification layer
                                               weights='imagenet')  # Use pre-trained weights on ImageNet

# Freeze the base model so its weights don't get updated during training
base_model.trainable = False

# Add your custom layers on top of the base model
model = tf.keras.Sequential([
    base_model,  # MobileNetV2 as the base model
    tf.keras.layers.GlobalAveragePooling2D(),  # Reduce dimensions to avoid overfitting
    tf.keras.layers.Dense(128, activation='relu'),  # Add a fully connected layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (real/fake)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=2,  # With a pre-trained model, fewer epochs can still give good results
    validation_data=val_generator
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
