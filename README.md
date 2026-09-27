# 基于 Transformer 的脑部 MRI 图像分割

本项目复现了论文《基于 Transformer 的脑部 MRI 图像分割方法在脑肿瘤检测中的应用》，在 BraTS2021 数据集上实现了三种医学图像分割模型，并提供了基于最优模型的辅助诊疗系统。

## 模型

| 模型 | 结构 | DSC | MIoU |
|------|------|-----|------|
| 线性神经网络 | 全连接网络 | 40.34% | 25.34% |
| U-Swin Transformer | Unet + Swin Transformer | 84.08% | 72.53% |
| TransBTS | 3D CNN + Transformer | 76.32% | 61.72% |

U-Swin Transformer 表现最好，被用于辅助诊疗系统。

## 目录结构

```
brain-tumor-segmentation/
├── models/            # 三个模型的网络定义
│   ├── linear_net.py  # 线性神经网络
│   ├── uswin.py       # U-Swin Transformer
│   ├── transbts.py    # TransBTS (3D)
│   └── swin_layers.py # Swin 基础组件
├── data/              # 数据加载与预处理
│   ├── preprocess.py  # nii 读取、切片、归一化
│   ├── dataset.py     # Dataset 类
│   └── transforms.py  # 数据增强
├── train/             # 训练脚本
│   ├── trainer.py     # 训练器
│   └── train.py       # 训练入口
├── utils/
│   └── metrics.py     # Dice / IoU 指标
├── scripts/
│   └── prepare_data.py # 数据预处理脚本
├── app/               # 辅助诊疗系统
│   ├── diagnosis.py   # GUI 界面
│   └── inference.py   # 命令行推理
└── checkpoints/       # 模型权重
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

从 [BraTS2021](http://braintumorsegmentation.org/) 下载数据集，解压后每个病例是一个文件夹，包含 t1/t2/t1ce/flair 四种模态和 seg 标签。

### 2D 数据（线性网络 / U-Swin）

```bash
python -m scripts.prepare_data --mode 2d --raw_dir /path/to/brats --out_dir data/2d
```

### 3D 数据（TransBTS）

```bash
python -m scripts.prepare_data --mode 3d --raw_dir /path/to/brats --out_dir data/3d
```

## 训练

```bash
# 线性网络
python -m train.train --model linear --data_dir data/2d --epochs 200

# U-Swin Transformer
python -m train.train --model uswin --data_dir data/2d --epochs 200

# TransBTS
python -m train.train --model transbts --data_dir data/3d --epochs 800 --batch_size 8
```

## 推理

```bash
python -m app.inference --image path/to/image.png --model checkpoints/best_model.pth --save result.png
```

## 辅助诊疗系统

```bash
python -m app.diagnosis
```

## 环境

- Python 3.12
- PyTorch 2.4.1
- GPU: NVIDIA Tesla V100

## 说明

- TransBTS 为 3D 模型，参数量大、计算资源要求高，论文中也指出其难以复现，实际部署建议使用 U-Swin Transformer。
- 标签含义：0=背景，1=水肿区，2=肿瘤核心，4=增强肿瘤。
