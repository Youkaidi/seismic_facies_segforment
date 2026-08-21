# ELL/VTPM组件消融

该目录实现论文中两个关键组件的2×2消融，四组实验除损失组件外使用完全相同的
U-Net、数据划分、初始化、优化器和训练参数。

| 变体 | 像素级损失 | VTPM约束 |
|---|---|---|
| `baseline` | 普通交叉熵（CE） | 否 |
| `ell` | 指数对数损失（ELL） | 否 |
| `vtpm` | 普通交叉熵（CE） | 是 |
| `full` | 指数对数损失（ELL） | 是 |

ELL默认使用`w_Dice=0.8`、`w_CE=0.2`、`gamma_Dice=gamma_CE=0.3`。
类别权重由训练划分按论文公式`sqrt(总像素数/该类像素数)`计算。VTPM由
softmax输出的相邻深度概率外积构造，因此从VTPM损失到网络输出的梯度链路完整可微。

完整实验：

```powershell
python -m ablation.run_ablation --resume
```

快速检查数据、前向传播及输出流程：

```powershell
python -m ablation.run_ablation `
  --variants baseline ell vtpm full `
  --epochs 1 --base-channels 8 `
  --max-train-sections 4 --max-validation-sections 2 --max-test-sections 2
```

默认结果保存到`ablation/outputs`：

- `configuration.json`：数据划分、类别频率、权重及全部超参数；
- `<variant>/best.pt`：验证集MIoU最高的权重；
- `<variant>/history.csv`：逐epoch训练记录；
- `<variant>/predictions/`：典型剖面PNG；
- `metrics.csv`：test1、test2及总体评价指标；
- `summary.json`：结构化完整结果。

正式实验默认训练60个epoch、批大小为2、Adam初始学习率为`1e-4`。连续3个
epoch训练损失没有改善时，学习率乘以0.2。`--resume`可以从每组的`last.pt`
继续运行。训练和验证剖面按项目原设置缩放为`256×192`，test1/test2仍在原始
分辨率上推理和计算指标；可以通过`--train-height`和`--train-width`修改。
