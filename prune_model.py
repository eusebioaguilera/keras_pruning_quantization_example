import numpy as np
import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tempfile

model = keras.models.load_model("model.h5")

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1  # 10% of training set will be used for validation set.

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
    'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                         final_sparsity=0.80,
                                         begin_step=0,
                                         end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model_for_pruning.summary()

logdir = tempfile.mkdtemp()

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(train_images,
                      train_labels,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_split=validation_split,
                      callbacks=callbacks)

_, model_for_pruning_accuracy = model_for_pruning.evaluate(test_images,
                                                           test_labels,
                                                           verbose=0)

print('Pruned test accuracy:', model_for_pruning_accuracy)

# Make model pruning by applying strip pruning
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

pruned_keras_file = "model_pruned.h5"
tf.keras.models.save_model(model_for_export,
                           pruned_keras_file,
                           include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

# ZIP model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

pruned_tflite_file = 'model_pruned.tflite'

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)
