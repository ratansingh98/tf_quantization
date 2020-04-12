import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
import os
quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
model = load_model('model.h5')
model.summary()
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# convet quantize model to tflite
quant_converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = quant_converter.convert()

# Convert normal model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Measure sizes of models.
tf_file = 'model.tflite'
quant_tffile = 'quant.tflite'

# write files
with open(quant_tffile, 'wb') as f:
  f.write(quantized_tflite_model)

with open(tf_file, 'wb') as f:
  f.write(tflite_model)

print("TF lite model in Mb:", os.path.getsize(tf_file) / float(2**20))
print("Quantized TF lite model in Mb:", os.path.getsize(quant_tffile) / float(2**20))