# MvHeatDET 代码库讲解文档

## 1. 项目是什么

MvHeatDET 是一个基于事件相机的目标检测项目，核心思路是：

1. 用 MvHeat 主干提取事件帧特征（MoE 热传导建模）。
2. 用 RT-DETR 风格的 Transformer 解码器做检测预测。
3. 用 Hungarian matching + 组合损失进行端到端训练。
4. 用 COCO 风格数据与评估流程进行训练和验证。

这个仓库不是“单脚本硬编码”风格，而是“配置驱动 + 注册器动态构建”风格。

---

## 2. 目录总览

```text
MvHeatDET/
├── README.md / README_zh-CN.md      # 论文与快速开始
├── configs/                         # 所有训练配置
│   ├── runtime.yml                  # 运行时通用设置
│   ├── dataset/                     # 数据集路径与类别设置
│   └── evheat/                      # 主模型配置（含 include）
├── src/
│   ├── core/                        # 注册器与 YAML 配置系统
│   ├── data/                        # 数据集、增强、dataloader
│   ├── solver/                      # 训练/验证循环
│   ├── zoo/                         # 模型、解码器、损失、后处理
│   ├── optim/                       # 优化器、EMA、AMP
│   └── misc/                        # 分布式与日志工具
├── tools/train.py                   # 训练/测试统一入口
├── train.sh / test.sh               # 常用命令示例
└── requirements.txt
```

---

## 3. 从命令到训练的调用链

以单卡训练命令为例：

```bash
python tools/train.py -c configs/evheat/MvHeatDET.yml
```

实际执行顺序可以理解为：

1. 入口脚本 `tools/train.py` 解析参数（config、resume、test-only 等）。
2. 构建 `YAMLConfig`，加载 `configs/evheat/MvHeatDET.yml` 及其 include 的子配置。
3. 读取 task 字段（这里是 detection），选择 `DetSolver`。
4. `DetSolver.fit()` 调用 `train_one_epoch()` 循环训练并周期评估。
5. 训练中使用：
   - model（DETR + MvHeat_DET + RTDETRTransformer）
   - criterion（SetCriterion）
   - postprocessor（RTDETRPostProcessor）
   - optimizer / lr_scheduler / ema / scaler

测试模式只需在命令里加 `-r checkpoint --test-only`，流程会走 `DetSolver.val()`。

---

## 4. 配置系统（本仓库最关键）

### 4.1 include 机制

主配置 `configs/evheat/MvHeatDET.yml` 通过 `__include__` 引入多个子配置：

- dataset 路径与类别
- runtime 通用开关
- dataloader 与增广
- optimizer 与调度器
- 模型结构

好处：不同实验可以复用公共片段，只改局部配置。

### 4.2 注册器 + 动态构建

`src/core/yaml_utils.py` 中的 `register/create` 机制会把 YAML 中的类型名映射为 Python 类并实例化。

比如：

- `model: DETR`
- `DETR.encoder: MvHeat_DET`
- `DETR.decoder: RTDETRTransformer`

会自动按依赖创建对象。

### 4.3 覆盖优先级

后加载配置中的同名字段会覆盖前面的值。常见场景：

- `runtime.yml` 给默认值
- `optimizer.yml` 再覆盖与训练策略相关字段（例如 `use_ema`、`find_unused_parameters`）

---

## 5. 模型部分如何看

### 5.1 外层封装：DETR

`src/zoo/rtdetr/rtdetr.py` 里的 `DETR` 类结构很直接：

1. `encoder(x)` 取特征。
2. `decoder(features, targets)` 输出预测。

### 5.2 编码器：MvHeat_DET

`src/zoo/evheat/vHeat_MoE.py` 中的 `MvHeat_DET` 是主干网络，关键配置：

- `depths`: 每个 stage 的 block 数
- `dims`: 每个 stage 通道数
- `img_size`, `patch_size`: 输入与下采样尺度
- `drop_path_rate`: 随机深度

其 forward 大致是：

1. Stem embedding
2. 多个 stage 的 HeatBlock + downsample
3. 输出最后一级特征图给解码器

### 5.3 解码器：RTDETRTransformer

`src/zoo/rtdetr/rtdetr_decoder.py` 负责：

1. 输入特征投影到统一 hidden dim。
2. 生成 encoder anchors 并筛选 top-k query。
3. 多层 decoder 迭代 refine 分类与框。
4. 输出 `pred_logits` 与 `pred_boxes`（以及训练时的辅助输出）。

---

## 6. 损失与后处理

### 6.1 损失：SetCriterion

`src/zoo/rtdetr/rtdetr_criterion.py`：

1. 先 Hungarian 匹配预测框和 GT。
2. 再算分类损失与框损失。

当前配置常见组合：

- `loss_vfl`
- `loss_bbox`
- `loss_giou`

权重在配置里由 `weight_dict` 控制。

### 6.2 后处理：RTDETRPostProcessor

`src/zoo/rtdetr/rtdetr_postprocessor.py`：

1. 由 `cxcywh` 转成 `xyxy`。
2. 根据原图尺寸反归一化到像素坐标。
3. 用 top-k 筛选输出（默认 focal 分支逻辑）。

---

## 7. 数据流

### 7.1 数据集

`src/data/coco/coco_dataset.py` 使用 `torchvision.datasets.CocoDetection`，并做：

- bbox 从 COCO 格式转为 `xyxy`
- 可选类别 remap
- 变换后写入 `target` 字段（boxes/labels/area/iscrowd 等）

### 7.2 增广

`src/data/transforms.py` 提供 Compose 与各类 v2 transform 注册。

训练配置中常见顺序：

1. 光度扰动
2. ZoomOut
3. IoU Crop
4. 水平翻转
5. Resize 到 640
6. Tensor 化与 dtype 转换
7. 框格式转 `cxcywh` 并归一化

### 7.3 collate

`src/data/dataloader.py` 的默认 `default_collate_fn`：

- 图像拼成 batch tensor
- 标注保持 list[dict]

这和 DETR 风格训练兼容。

---

## 8. 训练工程化细节

### 8.1 分布式

`src/misc/dist.py` 负责：

- `init_distributed()` 初始化进程组
- DDP 包装模型
- 分布式 sampler 包装 dataloader
- 多卡下 loss 字典归约

单卡场景如果没有分布式环境变量，会自动退化为普通模式。

### 8.2 EMA 与 AMP

- EMA 在 `src/optim/ema.py`
- 优化器与 lr scheduler 注册在 `src/optim/optim.py`
- AMP 通过配置中的 `use_amp/scaler` 打开

---

## 9. 你最常改的地方（实战）

### 9.1 改数据路径

改 `configs/dataset/EvDET200K_detection.yml`：

- `train_dataloader.dataset.img_folder`
- `train_dataloader.dataset.ann_file`
- `val_dataloader.dataset.img_folder`
- `val_dataloader.dataset.ann_file`

### 9.2 改模型规模

改 `configs/evheat/include/mvheatdet.yml`：

- `MvHeat_DET.depths`
- `MvHeat_DET.dims`
- `MvHeat_DET.img_size`
- `RTDETRTransformer.num_queries`

### 9.3 改训练策略

改 `configs/evheat/include/optimizer.yml`：

- `epoches`
- `optimizer.lr`
- `weight_decay`
- `use_ema`
- `clip_max_norm`

### 9.4 改输入尺寸

要同时改两处：

1. dataloader 的 Resize（如 640 -> 512）
2. 模型中的 `img_size` 和 decoder 的 `eval_spatial_size`

否则容易出现 shape/锚点不一致。

---

## 10. 常见问题排查

1. 训练一启动就报找不到数据：
   - 先检查 `EvDET200K_detection.yml` 路径是否绝对正确。

2. 多卡启动挂住：
   - 检查 `torchrun` 参数与环境变量，必要时按 README 加 `NCCL_P2P_DISABLE=1`。

3. 显存不够：
   - 先减 `batch_size`。
   - 再考虑降低输入尺寸或模型深度。

4. 评估指标异常低：
   - 检查类别 id 是否与标注一致。
   - 检查框格式转换是否符合 `cxcywh` 归一化约定。

---

## 11. 最小可跑步骤（建议）

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置数据路径（改 `configs/dataset/EvDET200K_detection.yml`）。

3. 先跑一次验证（验证权重加载和数据读取）：

```bash
python tools/train.py -c configs/evheat/MvHeatDET.yml -r <你的权重路径> --test-only
```

4. 再开始训练：

```bash
python tools/train.py -c configs/evheat/MvHeatDET.yml
```

---

## 12. 一句话总结

这是一个“配置驱动的事件相机 DETR 检测框架”：你只要先理清 `configs` 组合关系，再把握 `tools/train.py -> solver -> model/criterion/postprocessor` 这条主链路，就能高效地训练、调参和改模型。
