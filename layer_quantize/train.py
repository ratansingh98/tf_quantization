import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
import os
from tensorflow import keras
import time

def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for i, test_image in enumerate(test_images):
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float16)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  accuracy = (prediction_digits == test_labels).mean()
  return accuracy

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

# evaluate models
time_quant = time.time()
quant_test_accuracy = evaluate_model(quantized_tflite_model)
print("Time take by quantized tf lite  model is ",time.time()-time_quant)


time_tfmodel = time.time()
test_accuracy = evaluate_model(tflite_model)
print("Time take by tf lite  model is ",time.time()-time_tfmodel)
print("\n")
print("Accuracy of quantized tf lite model is",quant_test_accuracy)
print("Accuracy of tf lite model is",test_accuracy)
