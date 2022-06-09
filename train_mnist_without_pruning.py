import tempfile
import os

import tensorflow as tf
import numpy as np

from tensorflow import keras

from model import model

# Load MNIST dataset

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Train the digit classification model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.summary()

model.fit(
    train_images,
    train_labels,
    epochs=4,
    validation_split=0.1,
)

_, baseline_model_accuracy = model.evaluate(test_images,
                                            test_labels,
                                            verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

keras_file = 'model.h5'
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)

