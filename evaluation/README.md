# 结果评估模块

本目录统一存放地震相分割结果的评价代码。

## 文件结构

- `segmentation_metrics.py`：PA、MCA、MIoU、FWIU和混淆矩阵。
- `geological_metrics.py`：垂向转移统计和地质合理性指标。
- `evaluate_geological_metrics.py`：批量评估`.npy`预测剖面的命令行入口。
- `evaluate_result2_png.py`：批量评估`Result2`中五个模型的RGB PNG结果。
- `evaluate_segmentation.py`：原有SGEMS结果评估脚本。
- `tests/`：地质合理性指标的单元测试。

## 地质合理性指标

- `reverse_transition_rate`：反向层序转移率，越低越好。
- `vtpm_frobenius_to_target`：预测VTPM与真实VTPM的Frobenius距离，越低越好。

## 运行评估

请在项目根目录执行：

```powershell
python -m evaluation.evaluate_geological_metrics `
  --prediction-dir .\runs_11\preds_test_npy `
  --label-volume .\data\f3_model\test1_labels.npy `
  --method-name current_runs_11 `
  --output-dir .\result\geological_metrics\runs_11_core
```

输出包括聚合指标CSV、逐剖面指标CSV、完整JSON、转移矩阵NPZ和VTPM对比图。

## 运行测试

```powershell
python -m unittest discover -s evaluation\tests -v
```

## 评估Result2中的PNG结果

```powershell
python -m evaluation.evaluate_result2_png
```

脚本只读取`对照`（Baseline）、`DGCNN`、`Segformer`、`U-Net`和`MC_Net`目录，不会读取
`test1_replace`、`test2_replace`或其他目录。结果保存到`Result2/evaluation/`。
