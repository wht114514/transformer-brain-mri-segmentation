# transformer-brain-mri-segmentation

> **Transformer-based Brain MRI Segmentation for Brain Tumor Detection**
> 基于 Transformer 的脑部 MRI 图像分割方法在脑肿瘤检测中的应用

[English](#english) · [中文](#中文)

![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c)
![Python](https://img.shields.io/badge/Python-3.12.5-3776ab)
![Dataset](https://img.shields.io/badge/Dataset-BraTS2021-1d9e75)
![License](https://img.shields.io/badge/License-MIT-green)

---

<a name="english"></a>

## English

### Overview

Brain tumors — especially gliomas — are characterized by high incidence, recurrence, disability and mortality rates. Accurate segmentation of tumor sub-regions on MRI is a prerequisite for surgical planning, radiotherapy targeting and treatment monitoring. However, tumor boundaries against normal brain tissue are often diffuse, and the tumor itself is heterogeneous (active cells, necrotic core, edema).

This project benchmarks **three network families on the BraTS2021 dataset** for multimodal brain MRI tumor segmentation:

1. **Linear neural network** — a deliberately simple baseline.
2. **U-Swin Transformer (U-ST)** — U-Net backbone whose encoder is replaced by Swin Transformer, i.e. `U-Net + Swin Transformer`.
3. **TransBTS** — `Transformer + 3D CNN`; a 3D CNN encoder produces compact feature maps, a Transformer layer models long-range global dependencies, and a 3D CNN decoder restores resolution.

Models are evaluated with **Dice Similarity Coefficient (DSC)** and **mean IoU (MIoU)**. The best model is then wrapped into a **computer-aided diagnosis (CAD) prototype** for 2D MRI inference.

### Key Results

| Model | Structure | DSC (%) | MIoU (%) |
|-------|-----------|--------:|---------:|
| Linear network | linear activation baseline | 40.34 | 25.34 |
| **U-Swin Transformer (U-ST)** | **U-Net + Swin Transformer** | **84.08** | **72.53** |
| TransBTS | Transformer + 3D CNN | 76.32 | 61.72 |

**Findings**

- The linear baseline is far from usable — 3D medical segmentation is a genuinely non-linear problem.
- **The 2D U-Net + Swin Transformer configuration gave the best result**, outperforming the heavier 3D Transformer model by **+7.76 DSC**.
- TransBTS clearly beats the linear baseline, but its larger parameter count and depth made faithful reproduction difficult within the available budget — a reminder that architectural sophistication does not automatically translate into better metrics under limited data and compute.

> Because of the above, **U-ST was selected as the model behind the CAD prototype.**

### Method

**Dataset — BraTS2021.** Each case ships as five volumes:

```
BraTS2021_00001_t1.nii.gz      T1-weighted (longitudinal relaxation)
BraTS2021_00001_t2.nii.gz      T2-weighted (transverse relaxation)
BraTS2021_00001_t1ce.nii.gz    T1 with contrast enhancement
BraTS2021_00001_flair.nii.gz   FLAIR (fluid-attenuated inversion recovery)
BraTS2021_00001_seg.nii.gz     ground-truth label
```

The label volume is **integer-encoded and non-contiguous**:

| Value | Region |
|------:|--------|
| 0 | background |
| 1 | tumor outer / edema region |
| 2 | tumor core |
| 4 | enhancing tumor |

**Preprocessing.**

- Slices are extracted from `t1` + `seg` using a **0.05 segmentation threshold** to build a new 2D dataset and its labels.
- Masks are produced and indexed into a `GrayList.txt` manifest for the training loader.
- Splits follow **5-fold cross-validation**.
- Augmentation: random flip, scaling, rotation, cropping.

**Training settings.**

| | Linear / U-ST | TransBTS |
|---|---|---|
| Input | 2D slices | 3D volumes (packed to `.pkl`) |
| Patch size | 64 × 64 × 64 | 64 × 64 × 64 |
| Optimizer | SGD | AMSGrad |
| Learning rate | 1e-4 | 1e-5 |
| Batch size | 4 | 8 |
| Epochs | 200 (U-ST) | 800 |
| Loss / metrics | DSC, mean IoU | DSC, mean IoU |

**Data volume.**

| Task | Train | Test |
|---|---|---|
| 2D (linear, U-ST) | 5,300 slices (65.63 MB), from 758 patient studies | 1,325 slices (16.44 MB) |
| 3D (TransBTS) | 785 volumes (101 GB) | 167 volumes (22.66 GB) |

### CAD Prototype

The serving module loads the best U-ST checkpoint locally and runs inference on user-selected 2D MRI slices (`推理` / `退出` entry screen, multi-image selection, results rendered back to the UI). Preloaded weights are used because on-line training is too time- and compute-intensive.

Measured inference latency (10 runs):

| Images | Average time |
|---:|---:|
| 1 | 1.227 s |
| 10 | 12.593 s |
| 50 | ≈ 60.3 s |
| 100 | ≈ 107.5 s |

Design constraints considered: no arbitrary interface calls (registration/login gated), no clinician access to patient records from the admin side, and inference artifacts cleared on exit to protect patient privacy.

### Repository Structure

<!-- 请按实际文件名核对后再上传 -->

```
.
├── README.md
├── requirements.txt
├── （待确认：线性神经网络分割 baseline）
├── （待确认：U-Swin Transformer / U-ST 模型与训练）
├── （待确认：TransBTS 模型与训练）
├── （待确认：数据预处理与切片脚本，输出 GrayList.txt）
├── （待确认：DSC / MIoU 评估脚本）
├── （待确认：辅助诊疗系统 GUI）
├── checkpoints/            # 预训练权重（不入库）
└── data/                   # BraTS2021（不入库，见 .gitignore）
```

### Environment

Verified environment for the reported experiments:

| Item | Value |
|---|---|
| CPU | Intel Xeon Platinum 8255C @ 2.50 GHz |
| GPU | NVIDIA Tesla V100-SXM2 32 GB × 8 |
| OS | CentOS Linux 7.8.2003 (Core) |
| Memory | 38 GB (2D tasks) / 304 GB (TransBTS) |
| Python | 3.12.5 |
| PyTorch | 2.4.1 |

```bash
pip install torch==2.4.1 torchvision
pip install nibabel numpy matplotlib tqdm
```

> The reported runs used eight V100s. Single-GPU users should expect substantially longer training and should reduce `batch_size` first.

### Quick Start

> Commands will be finalized once the scripts are confirmed.

```bash
# 1. prepare 2D slices and the GrayList.txt manifest
# 2. train the linear baseline
# 3. train U-Swin Transformer (U-ST)
# 4. train TransBTS
# 5. evaluate DSC / MIoU
# 6. launch the CAD prototype
```

### Notes & Limitations

Stated plainly in the original report, and worth repeating here:

- **No clinically deployable model was produced.** The study verified the *feasibility* of applying Transformer to medical image segmentation and exercised an end-to-end application path — nothing more.
- **TransBTS was hard to reproduce faithfully** given its complexity; its metrics are therefore a lower bound rather than a tuned result.
- Data scale is limited relative to the capacity of Transformer models, so overfitting remains a risk.
- Only the `t1` modality was used for the 2D experiments, so the comparison does not yet exploit the full four-modality input.
- Reported DSC/MIoU are aggregate figures; no per-region (WT / TC / ET) breakdown was produced.

**Possible next steps:** lighter-weight architectures, per-region metrics, full four-modality 2D input, and stronger baselines (nnU-Net, Swin-UNet with pretrained weights).

### Data & Ethics

BraTS2021 is distributed by the **MICCAI Brain Tumor Segmentation Challenge** and is subject to its own terms of use. It is **not** redistributed in this repository. Please obtain the dataset through the official channel and comply with the challenge's data agreement. No patient-identifiable information is contained in this repository.

### Citation

```bibtex
@techreport{liu2024transformerbrains,
  title       = {基于 Transformer 的脑部 MRI 图像分割方法在脑肿瘤检测中的应用
                 (Transformer-based Brain MRI Segmentation for Brain Tumor Detection)},
  author      = {刘翔 and 王浩田},
  institution = {甘肃省庆阳第一中学},
  year        = {2024},
  note        = {腾讯"犀牛鸟中学科学人才培养计划"科研实践结题报告.
                 中学指导老师：李娟、梁庭嘉；高校指导老师：王连生}
}
```

---

<a name="中文"></a>

## 中文

### 项目简介

脑部肿瘤（尤其是脑胶质瘤）具有高发病率、高复发率、高致残率和高致死率的特点。在 MRI 上精确分割肿瘤各子区域，是手术方案制定、放疗靶区勾画和治疗效果监测的前提。但脑肿瘤与正常脑组织的边界常常模糊，且肿瘤内部结构复杂（活性细胞、坏死核心、水肿区域）。

本项目在 **BraTS2021 数据集**上对三类网络结构进行脑部 MRI 肿瘤分割的对比实验：

1. **线性神经网络** —— 刻意保留的简单基线。
2. **U-Swin Transformer（U-ST）** —— 用 Swin Transformer 替换 U-Net 编码器，即 `U-Net + Swin Transformer`。
3. **TransBTS** —— `Transformer + 3D CNN`。3D CNN 生成紧凑特征图以捕捉空间与深度信息，Transformer Layer 建模全局长距离依赖，再由 3D CNN 解码器恢复分辨率。

评价指标为 **Dice 系数（DSC）** 与 **平均 IoU（MIoU）**。最终以表现最好的模型为核心，搭建了一套面向 2D MRI 推理的**辅助诊疗系统原型**。

### 实验结果

| 模型 | 结构 | DSC (%) | MIoU (%) |
|------|------|--------:|---------:|
| 线性神经网络 | 线性激活基线 | 40.34 | 25.34 |
| **U-Swin Transformer（U-ST）** | **U-Net + Swin Transformer** | **84.08** | **72.53** |
| TransBTS | Transformer + 3D CNN | 76.32 | 61.72 |

**结论**

- 线性基线远达不到可用水平 —— 3D 医学影像分割本质上是非线性的问题。
- **2D 的 U-Net + Swin Transformer 取得了最好结果**，比更重的 3D Transformer 模型高出 **7.76 个 DSC 点**。
- TransBTS 明显优于线性基线，但其参数量与深度较大，在有限预算下难以忠实复现 —— 这提醒我们：结构更精巧，并不自动等于指标更好。

> 因此，**最终选用 U-ST 作为辅助诊疗系统背后的模型**。

### 方法

**数据集 BraTS2021。** 每个病例包含 5 个文件：

```
BraTS2021_00001_t1.nii.gz      T1 加权（纵向弛豫）
BraTS2021_00001_t2.nii.gz      T2 加权（横向弛豫）
BraTS2021_00001_t1ce.nii.gz    T1 增强成像
BraTS2021_00001_flair.nii.gz   液体衰减反转恢复
BraTS2021_00001_seg.nii.gz     标签
```

标签为**非连续整数编码**：

| 取值 | 区域 |
|-----:|------|
| 0 | 背景 |
| 1 | 肿瘤外液区（水肿） |
| 2 | 肿瘤核心区 |
| 4 | 增强肿瘤区域 |

**数据预处理**

- 以 `t1` + `seg` 为输入，按**分割阈值 0.05** 切片，生成新的 2D 数据集及对应标签。
- 生成 masks 并保存索引文件 `GrayList.txt`，供训练时加载。
- 按**五折交叉验证**划分。
- 数据增强：随机翻转、缩放、旋转、裁剪。

**训练设置**

| | 线性 / U-ST | TransBTS |
|---|---|---|
| 输入 | 2D 切片 | 3D 体数据（打包为 `.pkl`） |
| Patch 大小 | 64 × 64 × 64 | 64 × 64 × 64 |
| 优化器 | SGD | AMSGrad |
| 学习率 | 1e-4 | 1e-5 |
| Batch size | 4 | 8 |
| 训练轮数 | 200（U-ST） | 800 |
| 损失 / 指标 | DSC、mean IoU | DSC、mean IoU |

**数据量**

| 任务 | 训练集 | 测试集 |
|---|---|---|
| 2D（线性、U-ST） | 5,300 项（65.63 MB），来自 758 例病人 | 1,325 项（16.44 MB） |
| 3D（TransBTS） | 785 项（101 GB） | 167 项（22.66 GB） |

### 辅助诊疗系统

推理模块在本地加载最优 U-ST 权重，对用户选择的 2D MRI 切片做分割推理（入口界面为「推理 / 退出」，支持多图选择，结果回传前端显示）。使用预训练权重而非常规在线训练，是因为在线训练耗时且耗算力。

实测推理耗时（10 轮）：

| 图片数 | 平均耗时 |
|---:|---:|
| 1 | 1.227 s |
| 10 | 12.593 s |
| 50 | 约 60.3 s |
| 100 | 约 107.5 s |

设计时考虑过的约束：接口不可被任意调用（注册登录保护）、后台管理方无权查看患者检测记录、关闭软件后删除推理信息以保护患者隐私。

### 目录结构

<!-- 请按实际文件名核对后再上传 -->

```
.
├── README.md
├── requirements.txt
├── （待确认：线性神经网络分割 baseline）
├── （待确认：U-Swin Transformer / U-ST 模型与训练）
├── （待确认：TransBTS 模型与训练）
├── （待确认：数据预处理与切片脚本，输出 GrayList.txt）
├── （待确认：DSC / MIoU 评估脚本）
├── （待确认：辅助诊疗系统 GUI）
├── checkpoints/            # 预训练权重（不入库）
└── data/                   # BraTS2021（不入库，见 .gitignore）
```

### 环境依赖

实验所验证的环境：

| 项目 | 版本 |
|---|---|
| CPU | Intel Xeon Platinum 8255C @ 2.50 GHz |
| GPU | NVIDIA Tesla V100-SXM2 32 GB × 8 |
| 操作系统 | CentOS Linux 7.8.2003 (Core) |
| 内存 | 38 G（2D 任务）/ 304 G（TransBTS） |
| Python | 3.12.5 |
| PyTorch | 2.4.1 |

```bash
pip install torch==2.4.1 torchvision
pip install nibabel numpy matplotlib tqdm
```

> 原始实验使用了 8 张 V100。单卡用户训练时间会显著变长，建议优先下调 `batch_size`。

### 快速开始

> 命令待脚本文件名确认后补全。

```bash
# 1. 生成 2D 切片与 GrayList.txt 索引
# 2. 训练线性网络基线
# 3. 训练 U-Swin Transformer（U-ST）
# 4. 训练 TransBTS
# 5. 评估 DSC / MIoU
# 6. 启动辅助诊疗系统
```

### 局限与说明

以下是原报告中已明确写出的内容，在此如实保留：

- **本项目并未训练出符合实际临床诊断标准的模型。** 研究验证的是 Transformer 用于医疗影像分割的**可行性**，并实践了一条完整的应用落地路径，仅此而已。
- **TransBTS 因模型复杂而难以复现**，其指标应视为下界而非调优后的结果。
- 数据规模相对 Transformer 的容量仍然有限，存在过拟合风险。
- 2D 实验**仅使用了 `t1` 单模态**，尚未用满四模态输入。
- 报告的是整体 DSC / MIoU，**未给出按区域（WT / TC / ET）的细分指标**。

**可继续推进的方向：** 更轻量的网络结构、按区域的细分指标、四模态完整输入、引入更强基线（nnU-Net、带预训练权重的 Swin-UNet）。

### 数据与伦理

BraTS2021 由 **MICCAI 脑肿瘤分割挑战赛（BraTS Challenge）**发布，受其自身使用条款约束，**本仓库不重新分发该数据集**。请通过官方渠道获取并遵守挑战赛的数据使用协议。本仓库不包含任何可识别患者身份的信息。

### 引用

```bibtex
@techreport{liu2024transformerbrains,
  title       = {基于 Transformer 的脑部 MRI 图像分割方法在脑肿瘤检测中的应用
                 (Transformer-based Brain MRI Segmentation for Brain Tumor Detection)},
  author      = {刘翔 and 王浩田},
  institution = {甘肃省庆阳第一中学},
  year        = {2024},
  note        = {腾讯"犀牛鸟中学科学人才培养计划"科研实践结题报告.
                 中学指导老师：李娟、梁庭嘉；高校指导老师：王连生}
}
```

---

## License

MIT License
