import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
import os
from tensorflow import keras

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  tfmot.quantization.keras.quantize_annotate_layer(keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu)),
  keras.layers.BatchNormalization(),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.BatchNormalization(),
  keras.layers.Activation('relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()
# Compile the digit classification model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




quantize_model = tfmot.quantization.keras.quantize_apply(model)

# `quantize_model` requires a recompile.
quantize_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

quantize_model.summary()


# train normal model
model.fit(
  train_images,
  train_labels,
  batch_size=64,
  epochs=10,
  validation_split=0.2,
)


# `quantize_model` requires a recompile.
quantize_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train quantize normal model
quantize_model.fit(
  train_images,
  train_labels,
  batch_size=64,
  epochs=10,
  validation_split=0.2,
)


#convert quantize model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(quantize_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

# convert normal model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Measure sizes of models.
tf_file = 'model.tflite'
quant_file = 'quant.tflite'

# Write Files
with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

with open(tf_file, 'wb') as f:
  f.write(tflite_model)

#compare file size
print("Float model in Mb:", os.path.getsize(tf_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))
