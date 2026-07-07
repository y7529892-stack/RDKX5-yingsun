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
