# RDKX5-yingsun
## Deployment

This repository provides:
- Trained ONNX model (for inference)
- Edge deployment code for RDK X5 (OpenExplorer toolchain compatible)

The project is designed for direct deployment on RDK X5 after environment setup.

## Usage

1. Prepare RDK X5 environment (OpenExplorer SDK installed)
2. Deploy ONNX model to BPU via toolchain conversion
3. Run inference script:

```bash
python main.py

## Model Weights

The optimized deployment model for RDK X5 is provided as a compiled binary file due to its large size.

You can download the model from the following link:

- Model file: `new_model_x5.bin`  
- Download link: https://pan.baidu.com/s/1FfU0HGq-XxnASzqQyjPtaA  
- Extraction code: 8bsp

This model is used for edge inference on the RDK X5 BPU and is compatible with the deployed inference pipeline in this repository.
