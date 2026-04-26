# 配置目录说明

当前仓库已经整理成 FRED 专用配置，不再保留原始 EvDET200K 配置入口。

## 推荐入口

训练和测试统一使用：

```bash
python tools/train.py -c configs/fred_complete.yml
```

`configs/fred_complete.yml` 是唯一主入口，它负责声明任务类别，并 include 下面几个子配置：

- `runtime.yml`：AMP、EMA、SyncBN 等通用运行开关。
- `evheat/include/train_dataloader.yml`：FRED train/test 路径、序列级采样、图像增强和 batch 参数。
- `evheat/include/optimizer.yml`：训练轮数、学习率、优化器和学习率调度。
- `evheat/include/mvheatdet_fred.yml`：FRED 版轻量模型、decoder、loss 和后处理参数。

## 常改位置

- 改数据路径：`evheat/include/train_dataloader.yml`。
- 改训练集/测试集使用比例：`SequenceDatasetSubset.ratio` 或 `max_sequences`。
- 改 batch size：`train_dataloader.batch_size` 和 `val_dataloader.batch_size`。
- 改模型规模：`evheat/include/mvheatdet_fred.yml` 中的 `depths`、`dims`、`num_decoder_layers`。
- 改训练轮数和学习率：`evheat/include/optimizer.yml`。
