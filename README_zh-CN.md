# MvHeatDET-FRED

[English](./README.md) | [简体中文](./README_zh-CN.md)

`MvHeatDET-FRED` 是将 `MvHeatDET` 适配到 `FRED` 事件相机无人机检测数据集上的版本。这个仓库保留了原始 MvHeatDET 的检测框架，在数据侧接入了面向 FRED 的 COCO 风格数据流程，并提供了数据转换、训练和测试所需的入口脚本。

如果你希望完成下面这些工作，这个仓库就是为此准备的：

- 将原始 FRED 序列转换为 COCO 标注格式
- 在单类别无人机检测任务上训练 MvHeatDET
- 使用同一套检测流程评估模型权重
- 理解 FRED 版本是如何接入原始代码框架的

## 仓库内容概览

当前仓库的核心训练链路包括：

- 主干网络 `MvHeat_DET`，位于 [src/zoo/evheat/vHeat_MoE.py](/e:/ChenChong/project/MvHeatDET-FRED/src/zoo/evheat/vHeat_MoE.py)
- `RT-DETR` 风格的解码器、匹配器、损失函数和后处理模块，位于 [src/zoo/rtdetr](/e:/ChenChong/project/MvHeatDET-FRED/src/zoo/rtdetr)
- 基于 YAML 的配置与注册机制，位于 [src/core](/e:/ChenChong/project/MvHeatDET-FRED/src/core)
- FRED 数据集加载器，位于 [src/data/fred_dataset.py](/e:/ChenChong/project/MvHeatDET-FRED/src/data/fred_dataset.py)
- 统一训练/测试入口 [tools/train.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/train.py)

相比原始面向 EvDET200K 的配置，这个 FRED 版本主要改动在于：

- 数据集类型改为 `FREDDetection`
- 类别数改为 `1`
- 数据路径与 COCO 标注文件改为 FRED 版本
- 新增 FRED 数据转换脚本，位于 [tools](/e:/ChenChong/project/MvHeatDET-FRED/tools)

## 项目结构

```text
MvHeatDET-FRED/
|-- configs/
|   |-- fred_complete.yml               # FRED 训练主配置
|   |-- dataset/FRED_detection.yml      # 仅数据集相关配置
|   `-- evheat/include/                 # 模型、优化器、dataloader 等子配置
|-- src/
|   |-- core/                           # YAML 配置系统与注册机制
|   |-- data/                           # 数据集、增强、dataloader
|   |-- solver/                         # 训练与验证流程
|   |-- zoo/                            # 主干、解码器、匹配器、损失
|   `-- misc/                           # 分布式、日志、可视化工具
|-- tools/
|   |-- train.py                        # 主训练/测试入口
|   |-- convert_fred_to_coco.py         # 单个 split 转换脚本
|   |-- convert_all_fred_to_coco.py     # train/test 一键转换脚本
|   `-- export_onnx.py                  # ONNX 导出辅助脚本
|-- run_fred_training.py                # 训练流程便捷脚本
|-- run_fred_training.sh                # Shell 便捷脚本
|-- test_fred_dataset.py                # 数据集自检脚本
`-- README.md                           # 英文版说明
```

## 代码阅读建议

如果你想尽快读懂这个仓库，推荐按下面的顺序看：

1. 从 [tools/train.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/train.py) 开始。
   它负责解析 `--config`、`--resume`、`--test-only`、`--amp` 等参数，然后构建 `YAMLConfig`，再分发到对应任务的 solver。
2. 阅读 [src/core/yaml_config.py](/e:/ChenChong/project/MvHeatDET-FRED/src/core/yaml_config.py)。
   这是整个配置驱动机制的核心，负责加载 YAML、合并 include，并按需实例化模型、dataloader、优化器、EMA 和 AMP 相关对象。
3. 阅读 [src/solver/solver.py](/e:/ChenChong/project/MvHeatDET-FRED/src/solver/solver.py) 和 [src/solver/det_solver.py](/e:/ChenChong/project/MvHeatDET-FRED/src/solver/det_solver.py)。
   这两部分定义了初始化、权重恢复、训练、验证和最佳模型保存逻辑。
4. 阅读 [src/data/fred_dataset.py](/e:/ChenChong/project/MvHeatDET-FRED/src/data/fred_dataset.py)。
   这里是 FRED 专用的数据集包装层，它建立在通用 COCO loader 之上。
5. 阅读 [configs/fred_complete.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/fred_complete.yml) 及其 include 的子配置。
   这部分最能直接反映 FRED 版本真实使用的训练配方。
6. 阅读 [tools/convert_fred_to_coco.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/convert_fred_to_coco.py)。
   它展示了原始 FRED 目录结构要求，以及 `coordinates.txt` 是如何被转成 COCO 标注的。

## FRED 版本的工作方式

### 1. 数据集目录格式

仓库预期原始 FRED 数据集目录大致如下：

```text
FRED/
|-- train/
|   `-- <sequence_name>/
|       |-- coordinates.txt
|       `-- Event/Frames/*.png
`-- test/
    `-- <sequence_name>/
        |-- coordinates.txt
        `-- Event/Frames/*.png
```

转换脚本会读取：

- `Event/Frames/*.png` 中的事件帧图像
- `coordinates.txt` 中的边界框标注

并生成：

- `coco_annotations/train.json`
- `coco_annotations/test.json`

### 2. 标注转换流程

[tools/convert_fred_to_coco.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/convert_fred_to_coco.py) 用于转换单个 split。它会：

- 遍历 `train` 或 `test` 下的所有序列
- 解析 `coordinates.txt` 的每一行
- 为每张事件帧生成一个 COCO `image` 条目
- 为匹配时间戳的目标生成 COCO `annotation` 条目
- 使用单一类别 `drone`，其 `category_id = 1`

[tools/convert_all_fred_to_coco.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/convert_all_fred_to_coco.py) 则是同时处理 train 和 test 的包装脚本。

### 3. 数据集加载器

[src/data/fred_dataset.py](/e:/ChenChong/project/MvHeatDET-FRED/src/data/fred_dataset.py) 继承了通用 COCO 数据集加载器，主要做了两件事：

- 根据 COCO `file_name` 正确拼接 FRED 图像路径
- 对空框样本做保护，确保空的 bounding box 张量形状始终为 `[0, 4]`

### 4. 训练主流程

实际训练入口是：

```bash
python tools/train.py -c configs/fred_complete.yml
```

内部主调用链路为：

`tools/train.py` -> `YAMLConfig` -> `DetSolver` -> `train_one_epoch` / `evaluate`

### 5. 模型配置

FRED 版本的模型配方位于 [configs/evheat/include/mvheatdet_fred.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/mvheatdet_fred.yml)，主要设置包括：

- backbone：`MvHeat_DET`
- decoder：`RTDETRTransformer`
- criterion：`SetCriterion`
- postprocessor：`RTDETRPostProcessor`
- `num_classes: 1`
- 输入尺寸：`640 x 640`
- 查询数：`100`
- decoder 层数：`6`

## 环境安装

仓库当前推荐环境如下：

```bash
conda create -n mvheatfred python=3.8
conda activate mvheatfred
pip install -r requirements.txt
```

[requirements.txt](/e:/ChenChong/project/MvHeatDET-FRED/requirements.txt) 中的主要依赖包括：

- `torch==2.0.1`
- `torchvision==0.15.2`
- `pycocotools`
- `timm`
- `einops`
- `transformers`
- `thop`

## 数据准备

### 方式一：一次性转换 train 和 test

```bash
cd tools
python convert_all_fred_to_coco.py
```

### 方式二：手动逐个 split 转换

```bash
python tools/convert_fred_to_coco.py \
  --fred-root /path/to/FRED \
  --output-dir /path/to/FRED/coco_annotations \
  --split train

python tools/convert_fred_to_coco.py \
  --fred-root /path/to/FRED \
  --output-dir /path/to/FRED/coco_annotations \
  --split test
```

转换完成后，请确认：

- `/path/to/FRED/coco_annotations/train.json` 已生成
- `/path/to/FRED/coco_annotations/test.json` 已生成

也可以运行下面的脚本进行简单检查：

```bash
python test_fred_dataset.py
```

## 训练

### 常规训练

```bash
python tools/train.py -c configs/fred_complete.yml
```

### 从断点恢复训练

```bash
python tools/train.py -c configs/fred_complete.yml -r /path/to/checkpoint.pth
```

### 启用 AMP

```bash
python tools/train.py -c configs/fred_complete.yml --amp
```

### 多卡训练

```bash
torchrun --nproc_per_node=2 tools/train.py -c configs/fred_complete.yml --amp
```

当前 FRED 配置使用的是：

- batch size：`6`
- 训练轮数：`80`
- 优化器：`AdamW`
- 学习率：`1e-4`
- EMA：已开启

## 测试与评估

```bash
python tools/train.py -c configs/fred_complete.yml -r /path/to/checkpoint.pth --test-only
```

评估使用的验证集 dataloader 定义在 [configs/evheat/include/train_dataloader.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/train_dataloader.yml) 中，对应的是 FRED 的 `test` split。

## 便捷脚本

仓库里还提供了两个辅助脚本：

- [run_fred_training.py](/e:/ChenChong/project/MvHeatDET-FRED/run_fred_training.py)
- [run_fred_training.sh](/e:/ChenChong/project/MvHeatDET-FRED/run_fred_training.sh)

示例用法：

```bash
python run_fred_training.py --prepare-data --train --data-path /mnt/data/cc/FRED
python run_fred_training.py --test --checkpoint /path/to/checkpoint.pth --data-path /mnt/data/cc/FRED
```

## 路径相关说明

当前仓库中有几处默认写死了 Linux 风格的绝对路径：

- [configs/fred_complete.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/fred_complete.yml)
- [configs/dataset/FRED_detection.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/dataset/FRED_detection.yml)
- [configs/evheat/include/train_dataloader.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/train_dataloader.yml)
- [run_fred_training.py](/e:/ChenChong/project/MvHeatDET-FRED/run_fred_training.py)
- [tools/convert_all_fred_to_coco.py](/e:/ChenChong/project/MvHeatDET-FRED/tools/convert_all_fred_to_coco.py)

默认路径为：

```text
/mnt/data/cc/FRED
/mnt/data/cc/FRED/coco_annotations
/mnt/data/cc/FRED_output/MvHeatDET
```

如果你的数据不在这些位置，训练前请先把对应路径改掉。

## 关键配置文件

- [configs/fred_complete.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/fred_complete.yml)
  FRED 主配置入口，聚合了数据集、运行时、dataloader、优化器和模型配置。
- [configs/dataset/FRED_detection.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/dataset/FRED_detection.yml)
  仅数据集相关的配置，使用 `FREDDetection`。
- [configs/evheat/include/train_dataloader.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/train_dataloader.yml)
  训练与验证增强策略，以及 dataloader 参数。
- [configs/evheat/include/optimizer.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/optimizer.yml)
  训练轮数、优化器、调度器、EMA、梯度裁剪等设置。
- [configs/evheat/include/mvheatdet_fred.yml](/e:/ChenChong/project/MvHeatDET-FRED/configs/evheat/include/mvheatdet_fred.yml)
  模型结构、检测损失和后处理相关配置。

## 常见问题

### 找不到标注文件

请先确认已经生成：

- `coco_annotations/train.json`
- `coco_annotations/test.json`

### 数据目录存在，但图像加载失败

请检查原始 FRED 目录中是否仍然包含：

- `train/<sequence>/Event/Frames/*.png`
- `test/<sequence>/Event/Frames/*.png`

COCO json 中保存的 `file_name` 相对路径格式类似于：

```text
train/<sequence>/Event/Frames/<frame>.png
```

### 训练启动后因为路径报错

这通常是配置文件或辅助脚本里仍然使用了写死路径。把 `/mnt/data/cc/FRED` 替换成你本地的数据根目录即可。

### 出现空框或 shape 相关的 batch 错误

仓库已经在下面两个位置对 FRED 的空框情况做了兼容处理：

- [src/data/fred_dataset.py](/e:/ChenChong/project/MvHeatDET-FRED/src/data/fred_dataset.py)
- [src/solver/det_engine.py](/e:/ChenChong/project/MvHeatDET-FRED/src/solver/det_engine.py)

如果你后续修改了数据集或增强流程，建议保留这部分兼容逻辑。

## 致谢

这个仓库基于原始的 MvHeatDET / RT-DETR 风格代码框架扩展而来，并在此基础上适配了 FRED 数据集的训练和评估流程。

## 引用

如果你的工作使用了原始 MvHeatDET 方法，请引用相关论文：

```bibtex
@misc{wang2024EvDET200K,
  title={Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset},
  author={Xiao Wang and Yu Jin and Wentao Wu and Wei Zhang and Lin Zhu and Bo Jiang and Yonghong Tian},
  year={2024},
  eprint={2412.06647},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2412.06647}
}
```
