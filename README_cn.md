# RDKX5-yingsun

## 部署说明

本仓库提供：

- 用于推理的训练后 ONNX 模型
- 基于 RDK X5 的边缘部署代码（兼容 OpenExplorer 工具链）

完成环境配置后，本项目可直接部署于 RDK X5 平台运行。

## 使用方法

1. 准备 RDK X5 开发环境（安装 OpenExplorer SDK）

2. 通过工具链转换将 ONNX 模型部署至 BPU

3. 运行推理脚本：

```bash
python main.py
## Model Weights

The optimized deployment model for RDK X5 is provided as a compiled binary file due to its large size.

You can download the model from the following link:

- Model file: `new_model_x5.bin`  
- Download link: https://pan.baidu.com/s/1FfU0HGq-XxnASzqQyjPtaA  
- Extraction code: 8bsp

This model is used for edge inference on the RDK X5 BPU and is compatible with the deployed inference pipeline in this repository.```
