<<<<<<< HEAD
import tensorflow as tf
import os
import sys
import io
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure stdout uses UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Define image size and batch size
img_height, img_width = 224, 224
batch_size = 32

# Create an ImageDataGenerator instance for data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values
    rotation_range=40,    # Augmentation: Rotate images
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Reserve 20% of the data for validation
)

# Load training data
train_data = datagen.flow_from_directory(
    'D:\\KULIAH\\Lomba\\Hackaton\\Hackaton\\trashnet',  # Replace with the path to your TrashNet folder
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_data = datagen.flow_from_directory(
    'D:\\KULIAH\\Lomba\\Hackaton\\Hackaton\\trashnet',  # Replace with the path to your TrashNet folder
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  # Output layer with the number of classes
])
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    epochs=100,  # You can adjust the number of epochs
    validation_data=validation_data
)
=======
import tensorflow as tf
import os
import sys
import io
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure stdout uses UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Define image size and batch size
img_height, img_width = 224, 224
batch_size = 32

# Create an ImageDataGenerator instance for data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values
    rotation_range=40,    # Augmentation: Rotate images
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Reserve 20% of the data for validation
)

# Load training data
train_data = datagen.flow_from_directory(
    'D:\\KULIAH\\Lomba\\Hackaton\\Hackaton\\trashnet',  # Replace with the path to your TrashNet folder
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_data = datagen.flow_from_directory(
    'D:\\KULIAH\\Lomba\\Hackaton\\Hackaton\\trashnet',  # Replace with the path to your TrashNet folder
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')  # Output layer with the number of classes
])
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    epochs=100,  # You can adjust the number of epochs
    validation_data=validation_data
)
>>>>>>> 6743dd3 (first commit)
model.save ('recycle_model.keras')