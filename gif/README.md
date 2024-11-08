# GIF 超分辨率工具

使用 Real-ESRGAN 提升 GIF 图片分辨率的命令行工具。

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 下载模型：
   - 创建 `models` 目录
   - 从 [Real-ESRGAN 发布页](https://github.com/xinntao/Real-ESRGAN/releases) 下载模型文件
   - 将下载的模型文件放入 `models` 目录

## 使用方法

### 基本命令

```bash
python gif.py upscale <gif文件路径> -s <放大倍数> --model <模型名称>
```

### 参数说明
- `<gif文件路径>`: 输入的 GIF 文件路径
- `-s`, `--scale`: 放大倍数（2 或 4）
- `--model`: 使用的模型名称
- `--output`: (可选) 指定输出文件路径

### 示例
- 使用 anime 模型将 GIF 放大 4 倍
```bash
python gif.py upscale input.gif -s 4 --model realesrgan-x4plus-anime

# 指定输出路径
python gif.py upscale input.gif -s 4 --model realesrgan-x4plus-anime --output result.gif
```

### 支持的模型
- realesrgan-x4plus
- realesrgan-x4plus-anime
- realesrnet-x4plus