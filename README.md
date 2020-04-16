# Tensorflow Quantization
Here i have illustrated and compared speed and size affeact on quantization and model reduction.

## Layer Quantization

It is Post-training quantization conversion technique that can reduce model size while also improving CPU and hardware accelerator latency, with little degradation in model accuracy. 

|S No.|  Name | Train Accuracy  | Test Accuracy  |   Model Size|
|---|---|---|---|---|
| 1  | Float Model |0.9966   |0.9800   |   84.8KB|
| 2 |  Quantized Model  |0.9966   |  0.9732 |  23.1KB |

## Model Quantization

Quantization works by reducing the precision of the numbers used to represent a model's parameters, which by default are 32-bit floating point numbers. This results in a smaller model size and faster computation.

This was test on i5 8600K and nivida gtx 1080.
|S No.|  Name |  Test Accuracy  |Speed  |   Model Size|
|---|---|---|---|---|
| 1  |  Normal Model |   0.9823 |   0.3968 secs|  272.8KB |
| 2 |  Quantized Model | 0.9816   | 1.366 secs |  23.4KB |
| 2 |   TFLite Model| 0.9823    |0.8510 secs |  83.4KB |

Speed depends on architecture and devices.
