# MvHeat-DET

## 项目简介

这是论文 **"Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset"** 的官方 PyTorch 实现。

- 论文链接：<https://arxiv.org/abs/2412.06647>
- 代码链接：<https://github.com/Event-AHU/OpenEvDET/tree/main/MvHeatDET>

作者单位包含安徽大学、鹏城实验室、北京理工大学、北京大学等。

---

## 摘要（中文翻译）

事件流目标检测是当前研究热点，在低照度、运动模糊和快速运动场景下表现出明显优势。现有方法常以脉冲神经网络、Transformer 或卷积网络为核心，但分别存在性能上限、计算开销较大或局部感受野受限等问题。

本文提出一种基于 **MoE（专家混合）热传导机制** 的事件目标检测算法，在精度与效率之间取得了更好的平衡。方法流程为：

1. 使用 stem 网络进行事件数据嵌入；
2. 通过提出的 MoE-HCO 模块进行特征建模（用不同专家模块模拟事件流中的“热传导”）；
3. 使用基于 IoU 的查询选择模块进行高效 token 提取；
4. 将 token 输入检测头得到最终检测结果。

此外，论文还提出了新的事件检测基准数据集 **EvDET200K**：

- 采集设备：Prophesee EVK4-HD 高分辨率事件相机
- 类别数：10 类
- 标注框数量：200,000
- 样本数：10,054
- 每段时长：2~5 秒

作者还给出了 15+ 种 SOTA 检测器的系统对比结果，便于后续研究公平比较。

---

## 方法与结果说明（README解读）

README 主要包含以下几部分：

1. **方法框架图**：展示 MvHeat-DET 整体网络结构。
2. **实验结果图**：展示在基准上的对比表现。
3. **数据可视化**：展示 EvDET200K 的样例。
4. **Quick Start**：环境安装、数据准备、训练测试命令。
5. **Checkpoint 下载**：提供预训练权重下载地址与配置说明。
6. **致谢与引用**：说明继承仓库与论文引用格式。

---

## 快速开始（中文翻译）

### 1. 环境安装

作者实验使用单张 RTX 4090 24G 进行训练和评测。

```bash
conda create -n mvheat python=3.8
conda activate mvheat
pip install -r requirements.txt
```

### 2. 数据准备

下载 EvDET200K 数据集（百度网盘或 Dropbox），然后修改数据集路径：

- 配置文件：`configs/dataset/EvDET200K_detection.yml`

### 3. 训练

单卡训练：

```bash
python tools/train.py -c configs/evheat/MvHeatDET.yml
```

多卡训练：

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/MvHeatDET.yml
```

### 4. 测试

单卡测试：

```bash
python tools/train.py -c configs/evheat/MvHeatDET.yml -r ckp/mvheatdet_input640_layers18_dim768.pth --test-only
```

多卡测试：

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/MvHeatDET.yml -r ckp/mvheatdet_input640_layers18_dim768.pth --test-only
```

---

## 预训练权重（中文翻译）

作者提供了在 EvDET200K 上训练的预训练模型（百度网盘 / Dropbox），对应配置如下：

- 输入尺寸：640
- Block 数量：(2, 2, 18, 2)
- 通道数：(96, 192, 384, 768)

---

## 致谢（中文翻译）

代码基于以下仓库扩展实现，感谢这些开源工作：

- vHeat: <https://github.com/MzeroMiko/vHeat>
- RT-DETR: <https://github.com/lyuwenyu/RT-DETR>

---

## 引用（中文翻译）

如果此工作对你的研究有帮助，请引用：

```bibtex
@misc{wang2024EvDET200K,
      title={Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset},
      author={Xiao Wang and Yu Jin and Wentao Wu and Wei Zhang and Lin Zhu and Bo Jiang and Yonghong Tian},
      year={2024},
      eprint={2412.06647},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.06647},
}
```
