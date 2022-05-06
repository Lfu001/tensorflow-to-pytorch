# TensorFlow to PyTorch
Convert TensorFlow model's weights  to PyTorch models's weights. (For EfficientNet v2)

## What is this?
  1. Build EfficientNet v2 model in TensorFlow. This model is pretrained with `ImageNet 21k` and then fine-tuned to `ImageNet 1k`.
  2. For each layer, get weights as NumPy `ndarray`.
  3. Build EfficientNet v2 model in PyTorch, overwrite weights by weights we get in STEP 2.

  ```mermaid
    flowchart LR
      subgraph TensorFlow
        tf_weights
      end
      subgraph PyTorch
        torch_weights
      end
      tf_weights(Weights) --> np(NumPy ndarray) --> torch_weights(Weights)
      style TensorFlow fill:#9adbc4,stroke:#709c8c,stroke-width:2px,color:#fff
      style PyTorch fill:#ff8d70,stroke:#ff6b45,stroke-width:2px,color:#fff
      style tf_weights fill:#013243,stroke:#013243,stroke-width:2px,color:#fff
      style torch_weights fill:#013243,stroke:#013243,stroke-width:2px,color:#fff
      style np fill:#4dabcf,stroke:#4d77cf,stroke-width:2px,color:#fff
      linkStyle 0,1 stroke:#0c0d0d,stroke-width:6px
  ```
