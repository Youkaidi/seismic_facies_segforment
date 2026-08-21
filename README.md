# MC-Net地震相分割

本项目实现融合指数对数损失（ELL）与垂向转移概率矩阵（VTPM）约束的地震相
分割方法，用于缓解类别不平衡并提高预测结果的垂向层序合理性。

## 项目结构

- `model/`：U-Net及相关模型、损失函数；
- `data_process/`：F3数据读取与预处理；
- `evaluation/`：分割指标、地质合理性指标和Result2 PNG批量评估；
- `ablation/`：ELL/VTPM的2×2组件消融训练、测试与报告；
- `Result2/`：DGCNN、SegFormer、U-Net、Baseline和MC-Net的典型剖面结果；
- `result/`：已有评价结果和可视化。

## 环境安装

建议使用Python 3.8或更高版本，并根据本机CUDA版本安装对应的PyTorch：

```powershell
pip install -r requirements.txt
```

## 数据准备

受GitHub单文件大小限制，原始F3、Parihaka数据和模型权重不包含在仓库中。将F3
数组放入`data/f3_model/`，目录和数组形状见[data/README.md](data/README.md)。

## 结果评价

重新评价`Result2`中的PNG结果：

```powershell
python -m evaluation.evaluate_result2_png
```

输出位于`Result2/evaluation/`，包括逐剖面、分测试区和总体指标。

## ELL/VTPM组件消融

四组配置分别为CE、ELL、CE+VTPM和ELL+VTPM：

```powershell
python -m ablation.run_ablation --resume
```

实验设计、参数和输出说明见[ablation/README.md](ablation/README.md)，已有实验报告见
[ablation/outputs/ablation_report.md](ablation/outputs/ablation_report.md)。

## 说明

仓库保留轻量级PNG、CSV和JSON结果，但忽略原始体数据、训练断点与权重。若需要
复现论文数值，应保持训练/验证划分、随机种子、损失权重和数据预处理配置一致。

