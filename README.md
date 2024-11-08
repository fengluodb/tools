# AI 工具集

这是一个实用的 AI 工具集合，包含多个独立的工具模块。

## 工具列表

### 1. GIF 超分辨率工具 (`/gif`)
使用 Real-ESRGAN 提升 GIF 图片分辨率的命令行工具。
- 支持 2x 和 4x 放大
- 支持多种 Real-ESRGAN 模型
- [详细说明](./gif/README.md)

## 环境要求

- Python 3.8+
- CUDA (可选，用于 GPU 加速)

## 快速开始

1. 克隆仓库：
```bash
git clone <repository-url>
cd tools
```

2. 选择需要使用的工具，进入对应目录：
```bash
cd gif  # 例如使用 GIF 工具
```

3. 按照各工具目录中的 README 说明进行安装和使用

## 目录结构

```
.
├── README.md
├── gif/
│   ├── README.md
│   ├── gif.py
│   └── models/
└── ... (其他工具)
```

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request！