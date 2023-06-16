import os
import cv2
# import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
# import scipy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Path Set
train_path = "data/gestures/train"
test_path = "data/gestures/test"
validate_path = "data/gestures/test"

train_batches = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, rotation_range=45, horizontal_flip=True).flow_from_directory(directory=train_path, target_size=(64, 64), batch_size=10, color_mode="grayscale", class_mode="categorical")
# print(f"train_batches: {train_batches}")
test_batches = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, rotation_range=45, horizontal_flip=True).flow_from_directory(directory=test_path, target_size=(64, 64), batch_size=10, color_mode="grayscale", class_mode="categorical")
# validate_batches = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, rotation_range=45, horizontal_flip=True).flow_from_directory(directory=validate_path, target_size=(64, 64), batch_size=10, color_mode="grayscale", class_mode="categorical")


imgs, labels = next(train_batches)

# Plotting images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(64, 64))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

print("Example images for reference on which our network would be trained")
plotImages(imgs)

# CNN Architecture
'''model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=(64, 64, 1)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=20, activation="softmax")
])'''

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=(64, 64, 1)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same"),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same"),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=150, activation="relu"),
    Dense(units=75, activation="relu"),
    Dense(units=20, activation="softmax")
])

model.summary()

# Compiling model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])


# setting checkpoint and using save best only model
checkpoint_filepath = 'weights.{epoch:02d}-{val_loss:.2f}.h5'

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(train_batches, epochs=100, verbose=2, validation_data=test_batches, steps_per_epoch=1800, callbacks=[model_checkpoint_callback])


'''
# Training model
model.fit(train_batches, epochs=100, verbose=2, validation_data=test_batches, steps_per_epoch=1800)
'''

# Save model
if os.path.isfile("model_1.h5") is False:
    model.save("model_1.h5")
