# train_emotion_model.py
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Load preprocessed data
data = np.load('educational_system/data/preprocessed_fer2013.npz')
train_images = data['train_images']
train_labels = to_categorical(data['train_labels'])
val_images = data['val_images']
val_labels = to_categorical(data['val_labels'])

# Normalize the images
train_images = train_images / 255.0
val_images = val_images / 255.0

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # Assuming 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Save the model
model.save('educational_system/models/emotion_model.h5')
