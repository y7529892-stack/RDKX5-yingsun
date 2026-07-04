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
