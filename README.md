# TensorFlow to PyTorch
Convert TensorFlow model's weights  to PyTorch models's weights. (For EfficientNet v2)

## What is this?
  1. Build EfficientNet v2 model in TensorFlow. This model is pretrained with `ImageNet 21k` and then fine-tuned to `ImageNet 1k`.
  2. For each layer, get weights as NumPy `ndarray`.
  3. Build EfficientNet v2 model in PyTorch, overwrite weights by weights we get in STEP 2.

  ```mermaid
    flowchart LR
      tf(TensorFlow) -- weights --> np(NumPy ndarray) -- weights --> torch(PyTorch)
  ```
