# MvHeatDET-FRED 代码导览

这份导览面向当前 FRED 事件相机无人机检测版本。仓库已经整理为 FRED 专用训练流程，推荐入口是 `configs/fred_complete.yml`。

## 训练入口

```bash
python tools/train.py -c configs/fred_complete.yml
```

执行链路：

1. `tools/train.py` 解析命令行参数。
2. `src/core/yaml_config.py` 加载 `configs/fred_complete.yml` 及其 include。
3. 根据 `task: detection` 创建 `DetSolver`。
4. `src/solver/det_solver.py` 负责训练、验证、保存 checkpoint。
5. `src/solver/det_engine.py` 执行单轮 train/evaluate。

## 配置系统

当前配置结构：

```text
configs/
|-- fred_complete.yml
|-- runtime.yml
`-- evheat/
    `-- include/
        |-- train_dataloader.yml
        |-- optimizer.yml
        `-- mvheatdet_fred.yml
```

常改文件：

- 数据路径、序列采样比例、batch size：`configs/evheat/include/train_dataloader.yml`。
- 学习率、训练轮数、EMA、scheduler：`configs/evheat/include/optimizer.yml`。
- 模型规模、decoder 层数、query 数量：`configs/evheat/include/mvheatdet_fred.yml`。
- 输出目录和总入口：`configs/fred_complete.yml`。

## 数据流程

FRED 原始目录需要先转为 COCO 格式：

```bash
python tools/convert_all_fred_to_coco.py --fred-root /mnt/data/cc/FRED
```

关键文件：

- `tools/convert_fred_to_coco.py`：转换单个 split。
- `tools/convert_all_fred_to_coco.py`：批量转换 train/test。
- `src/data/fred_dataset.py`：按 COCO 标注读取 FRED 图像。
- `src/data/dataloader.py`：包含 `SequenceDatasetSubset`，用于按序列采样而不是按单张图随机采样。

## 模型流程

模型由 `configs/evheat/include/mvheatdet_fred.yml` 组装：

- `MvHeat_DET`：事件帧图像主干网络。
- `RTDETRTransformer`：RT-DETR 风格检测 decoder。
- `SetCriterion`：分类和框回归损失。
- `RTDETRPostProcessor`：推理后处理和 top-k 选择。

## 验证与保存

训练时 `DetSolver.fit()` 每个 epoch 后会运行验证，并保存 checkpoint 到 `output_dir`。

常用命令：

```bash
python tools/train.py -c configs/fred_complete.yml
python tools/train.py -c configs/fred_complete.yml -r /path/to/checkpoint.pth
python tools/train.py -c configs/fred_complete.yml -r /path/to/checkpoint.pth --test-only
```

## 排查优先级

- AP 为 0：优先检查 COCO 标注和图像路径是否对应，尤其是 frame 文件名里的时间戳。
- 训练太慢：先调低 `SequenceDatasetSubset.ratio`，再考虑减小模型或 batch。
- 显存没吃满：优先增大 `batch_size`，但保持验证 batch 与训练 batch 分开观察。
- label 越界：检查 `num_classes: 1`、COCO `category_id`、`remap_mscoco_category` 是否一致。
